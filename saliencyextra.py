# importaciones
import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models, transforms, datasets

# correr coridgo
# python saliencyextra.py --weights chest_aid_densenet121.pth --test_dir ./test --out_dir ./saliency_metrics --t 1.171 --n_images 64 --deletion_fill blur --sufficiency_fill mean
# se puede elegir entre mean, black y blur para los rellenos de deletion


# obtiene el dispositivo de ejecucion gpu o cpu
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# fija la semilla aleatoria para reproducibilidad
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# construye la transformacion de prueba para imagenes
def build_test_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# carga el modelo densenet121 ajustado para clasificacion binaria
def load_densenet121_for_binary(weights_path: str, device=None):
    device = device or get_device()
    model = models.densenet121(weights=None)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, 1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# aplica la temperatura al logits para calibracion
def apply_temperature(logits: torch.Tensor, t_value: Optional[float]):
    if t_value is None:
        return logits
    t = float(max(0.05, min(10.0, t_value)))
    return logits / t


# predice probabilidades con el modelo aplicando temperatura si existe
@torch.no_grad()
def predict_prob(
    model: nn.Module, x: torch.Tensor, t_value: Optional[float], device=None
):
    device = device or get_device()
    model = model.to(device).eval()
    x = x.to(device)
    logits = model(x)
    logits = apply_temperature(logits, t_value)
    probs = torch.sigmoid(logits).squeeze(1)
    return probs.detach().cpu().numpy()


# convierte tensor normalizado a imagen uint8
def denorm_to_uint8(x_bchw: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=x_bchw.device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=x_bchw.device)[None, :, None, None]
    img = (
        (x_bchw * std + mean)
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    return (img * 255).astype(np.uint8)


# convierte imagen uint8 a tensor normalizado
def uint8_to_norm_tensor(arr_uint8: np.ndarray, device=None):
    device = device or get_device()
    img = Image.fromarray(arr_uint8)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(img).unsqueeze(0).to(device)


# obtiene la ultima capa convolucional de un modelo
def get_last_conv2d(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# clase para generar mapas de activacion gradcam
class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activ = None
        self.grads = None
        self.h_f = target_layer.register_forward_hook(self._on_fwd)
        self.h_b = target_layer.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, module, inp, out):
        self.activ = out.detach()

    def _on_bwd(self, module, gin, gout):
        self.grads = gout[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = 0):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activ).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def close(self):
        self.h_f.remove()
        self.h_b.remove()


# ajusta tamaño del mapa de calor a dimensiones de destino
def upsample_to(cam_1chw: torch.Tensor, target_hw: Tuple[int, int]):
    heat = F.interpolate(cam_1chw, size=target_hw, mode="bilinear", align_corners=False)
    heat = heat.squeeze().detach().cpu().numpy()
    return np.clip(heat, 0, 1)


# crea mascara binaria a partir de mapa de calor
def make_mask_from_heat(heat_01: np.ndarray, k_percent: float):
    h, w = heat_01.shape
    flat = heat_01.reshape(-1)
    n_keep = max(1, int(len(flat) * (k_percent / 100.0)))
    order = np.argsort(-flat)
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[order[:n_keep]] = 1.0
    return mask.reshape(h, w)


# crea imagen de relleno segun el modo especificado
def build_fill_image(arr_uint8: np.ndarray, fill_mode: str):
    h, w, _ = arr_uint8.shape
    if fill_mode == "black":
        base = np.zeros_like(arr_uint8, dtype=np.uint8)
    elif fill_mode == "mean":
        mean_rgb = arr_uint8.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        base = np.tile(mean_rgb[None, None, :], (h, w, 1))
    elif fill_mode == "blur":
        base = np.array(
            Image.fromarray(arr_uint8).filter(ImageFilter.GaussianBlur(radius=5))
        )
    else:
        mean_rgb = arr_uint8.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        base = np.tile(mean_rgb[None, None, :], (h, w, 1))
    return base


#  combina imagen original y relleno segun mascara y modo
def compose_with_fill(
    x_1chw: torch.Tensor, mask_hw: np.ndarray, fill_mode: str, keep: bool
):
    arr_orig = denorm_to_uint8(x_1chw)
    arr_fill = build_fill_image(arr_orig, fill_mode)
    m = mask_hw > 0.5
    out = arr_orig.copy()
    if keep:
        out[~m] = arr_fill[~m]
    else:
        out[m] = arr_fill[m]
    return uint8_to_norm_tensor(out, device=x_1chw.device)


# curvas de suficiencia y eliminación
@torch.no_grad()
def sufficiency_deletion_curves(
    model,
    x_1chw,
    heat_01,
    t_value: Optional[float],
    ks=(1, 3, 5, 10, 20, 30, 40, 50),
    sufficiency_fill: str = "mean",
    deletion_fill: str = "blur",
):
    ks_list, keep_list, del_list = [], [], []
    for k in ks:
        mask = make_mask_from_heat(heat_01, k)
        x_keep = compose_with_fill(x_1chw, mask, fill_mode=sufficiency_fill, keep=True)
        x_del = compose_with_fill(x_1chw, mask, fill_mode=deletion_fill, keep=False)
        p_keep = predict_prob(model, x_keep, t_value)[0]
        p_del = predict_prob(model, x_del, t_value)[0]
        ks_list.append(k)
        keep_list.append(p_keep)
        del_list.append(p_del)
    return ks_list, keep_list, del_list


# area bajo la curva por el metodo del trapecio
def auc_trapz(xs, ys):
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    return float(np.trapz(ys, xs) / (xs.max() - xs.min() + 1e-8))


# chequear correlación entre el mapa de calor original y el de un modelo aleatorio
def randomize_model_copy(model: nn.Module):
    import copy

    m = copy.deepcopy(model).cpu()

    def _reset(module):
        for child in module.children():
            _reset(child)
        if hasattr(module, "reset_parameters"):
            try:
                module.reset_parameters()
            except Exception:
                pass

    _reset(m)
    return m


# mapa de correlación entre dos mapas de calor
def map_correlation(a_01: np.ndarray, b_01: np.ndarray):
    af = a_01.flatten()
    bf = b_01.flatten()
    af = (af - af.mean()) / (af.std() + 1e-8)
    bf = (bf - bf.mean()) / (bf.std() + 1e-8)
    return float(np.clip((af * bf).mean(), -1, 1))


# datos construcción de loader de test
def build_test_loader(
    test_dir: str, batch_size: int = 8, img_size: int = 224, num_workers: int = 2
):
    tfm = build_test_transform(img_size)
    ds = datasets.ImageFolder(root=test_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader


# pipeline principal de análisis cuantitativo de grad-cam
def run_analysis(
    weights_path: str,
    test_dir: str,
    out_dir: str,
    t_value: Optional[float],
    img_size: int = 224,
    batch_size: int = 8,
    n_images: int = 32,
    ks: Tuple[int, ...] = (1, 3, 5, 10, 20, 30, 40, 50),
    sufficiency_fill: str = "mean",
    deletion_fill: str = "blur",
    seed: int = 42,
):

    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = get_device()

    model = load_densenet121_for_binary(weights_path, device=device)
    ds, loader = build_test_loader(test_dir, batch_size=batch_size, img_size=img_size)

    try:
        target_layer = get_last_conv2d(model.features)
    except AttributeError:
        target_layer = get_last_conv2d(model)
    cam_engine = GradCam(model, target_layer)

    count = 0
    suff_aucs, del_aucs, corrs = [], [], []
    ks_ref = None
    keep_sum = None
    del_sum = None

    for xb, yb in loader:
        xb = xb.to(device)
        for i in range(xb.size(0)):
            x1 = xb[i : i + 1]

            cam = cam_engine.generate(x1, class_idx=0)
            heat = upsample_to(cam, (x1.shape[2], x1.shape[3]))

            ks_list, keep_list, del_list = sufficiency_deletion_curves(
                model,
                x1,
                heat,
                t_value=t_value,
                ks=ks,
                sufficiency_fill=sufficiency_fill,
                deletion_fill=deletion_fill,
            )

            suff_aucs.append(auc_trapz(ks_list, keep_list))
            del_aucs.append(auc_trapz(ks_list, del_list))

            if ks_ref is None:
                ks_ref = ks_list
                keep_sum = np.zeros(len(ks_ref), dtype=np.float64)
                del_sum = np.zeros(len(ks_ref), dtype=np.float64)
            keep_sum += np.array(keep_list, dtype=np.float64)
            del_sum += np.array(del_list, dtype=np.float64)

            # sanity check por imagen correlación vs grad-cam de un modelo aleatorio
            m_rand = randomize_model_copy(model).to(device).eval()
            try:
                tl_rand = get_last_conv2d(m_rand.features)
            except AttributeError:
                tl_rand = get_last_conv2d(m_rand)
            cam_rand = GradCam(m_rand, tl_rand).generate(x1, class_idx=0)
            heat_rand = upsample_to(cam_rand, (x1.shape[2], x1.shape[3]))
            corrs.append(map_correlation(heat, heat_rand))

            count += 1
            if count >= n_images:
                break
        if count >= n_images:
            break

    cam_engine.close()

    suff_auc_mean = float(np.mean(suff_aucs)) if suff_aucs else None
    del_auc_mean = float(np.mean(del_aucs)) if del_aucs else None
    sanity_corr_mean = float(np.mean(corrs)) if corrs else None

    avg_keep_curve = (
        (keep_sum / max(1, count)).tolist() if keep_sum is not None else None
    )
    avg_del_curve = (del_sum / max(1, count)).tolist() if del_sum is not None else None

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "weights_path": weights_path,
            "test_dir": test_dir,
            "out_dir": out_dir,
            "t_value": t_value,
            "img_size": img_size,
            "batch_size": batch_size,
            "n_images": count,
            "ks": ks,
            "sufficiency_fill": sufficiency_fill,
            "deletion_fill": deletion_fill,
            "seed": seed,
        },
        "metrics": {
            "sufficiency_auc_mean": suff_auc_mean,
            "deletion_auc_mean": del_auc_mean,
            "sanity_corr_mean": sanity_corr_mean,
        },
        "curves": {
            "ks": ks_ref,
            "avg_keep_curve": avg_keep_curve,
            "avg_del_curve": avg_del_curve,
        },
    }

    with open(
        os.path.join(out_dir, "saliency_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if ks_ref is not None:
        csv_path = os.path.join(out_dir, "saliency_curves_avg.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("k_percent,prob_keep,prob_delete\n")
            for k, pk, pd in zip(ks_ref, avg_keep_curve, avg_del_curve):
                f.write(f"{k},{pk:.6f},{pd:.6f}\n")

    print(json.dumps(summary["metrics"], indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="analisis cuantitativo de grad-cam")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--t", type=float, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_images", type=int, default=32)
    p.add_argument("--ks", type=str, default="1,3,5,10,20,30,40,50")
    p.add_argument(
        "--sufficiency_fill",
        type=str,
        default="mean",
        choices=["mean", "black", "blur"],
    )
    p.add_argument(
        "--deletion_fill", type=str, default="blur", choices=["mean", "black", "blur"]
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    run_analysis(
        weights_path=args.weights,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        t_value=args.t,
        img_size=args.img_size,
        batch_size=args.batch_size,
        n_images=args.n_images,
        ks=ks,
        sufficiency_fill=args.sufficiency_fill,
        deletion_fill=args.deletion_fill,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
