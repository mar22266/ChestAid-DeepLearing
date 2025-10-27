# importaciones
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib.pyplot as plt
import gradio as gr

# configuración de pesos y transformaciones
weights_path = "chest_aid_densenet121.pth"
temp_path = "temperature_scaler.pth"
img_size = 224

# medias y desvios de imagenet para normalizar entradas
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# pipeline de evaluacion con redimension tensor y normalizacion
eval_tfms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

# seleccion de dispositivo usa cuda si esta disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# crea el modelo densenet121 con pesos de imagenet y clasificador para una salida
def build_model(num_classes: int = 1) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return model


# calibrador de temperatura para ajustar la confianza de los logits
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature.clamp(0.05, 10.0)
        return logits / t


# obtiene la ultima capa convolucional dentro de features
def get_last_conv_layer(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# grad cam minimal con hooks para activaciones y gradientes
class GradCAM:
    # guarda referencias del modelo y la capa objetivo y registra hooks
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_module.register_forward_hook(self._save_activation)
        self.bwd_hook = target_module.register_full_backward_hook(self._save_gradient)

    # almacena activaciones sin gradiente
    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    # almacena gradientes de salida sin gradiente
    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    # genera el mapa grad cam normalizado
    def generate(self, x: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# convierte un tensor normalizado a imagen numpy en rango cero uno
def tensor_to_numpy_image(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(imagenet_mean).view(3, 1, 1)
    std = torch.tensor(imagenet_std).view(3, 1, 1)
    img = img_t.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# crea superposicion del grad cam sobre la imagen original
def overlay_cam_on_image(
    img_t: torch.Tensor, cam_t: torch.Tensor, alpha: float = 0.35
) -> np.ndarray:
    base = tensor_to_numpy_image(img_t)
    heat = cam_t.squeeze().cpu().numpy()
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_uint8).resize(
        (base.shape[1], base.shape[0]), resample=Image.BILINEAR
    )
    heat_arr = np.array(heat_img) / 255.0
    cmap = plt.get_cmap("jet")
    heat_color = cmap(heat_arr)[..., :3]
    overlay = (1 - alpha) * base + alpha * heat_color
    overlay = overlay.clip(0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


# crea el modelo y lo mueve al dispositivo
model = build_model().to(device)
if not Path(weights_path).exists():
    raise FileNotFoundError(
        f"no se encontró '{weights_path}'. entrena el modelo y guarda los pesos en ese archivo."
    )
# carga estado de pesos desde disco
state = torch.load(weights_path, map_location=device)
model.load_state_dict(state)
model.eval()

# grad-cam
target_conv = get_last_conv_layer(model.features)
cam_engine = GradCAM(model, target_conv)

# crea el escalador de temperatura y lo mueve al dispositivo
temp_scaler = TemperatureScaler().to(device)
if Path(temp_path).exists():
    temp_state = torch.load(temp_path, map_location=device)
    temp_scaler.load_state_dict(temp_state)
    temperature_value = float(temp_scaler.temperature.data.item())
else:
    temperature_value = 1.0


# predice y explica una imagen con grad cam y calibracion
def predict_and_explain(
    image: Image.Image, threshold: float = 0.5, alpha: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if image is None:
        return None, None, {"error": "no se recibió ninguna imagen"}
    img_rgb = image.convert("RGB")
    x_t = eval_tfms(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_t)
        t = max(0.05, min(10.0, float(temperature_value)))
        logits = logits / t
        prob = torch.sigmoid(logits).item()

    cam = cam_engine.generate(x_t, class_idx=0)
    overlay = overlay_cam_on_image(x_t[0], cam[0], alpha=alpha)
    orig = (tensor_to_numpy_image(x_t[0]) * 255).astype(np.uint8)

    # arma diccionario informativo con resultados
    pred_lbl = int(prob >= threshold)
    pred_txt = "NEUMONÍA" if pred_lbl == 1 else "NORMAL"
    info = {
        "probabilidad_calibrada": round(prob, 4),
        "umbral": round(threshold, 3),
        "prediccion": pred_txt,
        "temperatura_usada": round(float(temperature_value), 4),
        "nota": "la probabilidad está calibrada con temperature scaling; grad-cam resalta regiones atencionales.",
    }
    return orig, overlay, info


# construye la interfaz de gradio para la demo
with gr.Blocks(title="Chest-Aid — Triage asistido por IA") as demo:
    gr.Markdown(
        """
        # chest-aid — triage asistido por ia
        herramienta para apoyar el triage radiológico inicial en rx de tórax (neumonía vs normal).
        sube una imagen, ajusta el umbral y observa la probabilidad calibrada y el grad-cam.
        """
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="radiografía de tórax", type="pil")
            thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="umbral de decisión")
            a = gr.Slider(
                0.0,
                1.0,
                value=0.35,
                step=0.05,
                label="alpha del grad-cam (transparencia)",
            )
            btn = gr.Button("evaluar")
            gr.Markdown(
                f"> temperatura calibrada detectada: **{temperature_value:.3f}** "
                f"({'archivo encontrado' if Path(temp_path).exists() else 'sin archivo, usando 1.0'})"
            )
        with gr.Column():
            img_orig = gr.Image(label="imagen original", interactive=False)
            img_cam = gr.Image(label="grad-cam (superposición)", interactive=False)
            info_out = gr.JSON(label="resultados")
    btn.click(
        predict_and_explain,
        inputs=[img_in, thr, a],
        outputs=[img_orig, img_cam, info_out],
    )

# lanza la demo en modo local con navegador
demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
