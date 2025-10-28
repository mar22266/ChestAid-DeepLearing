# Chest-Aid: Triage asistido por IA para neumonía en rayos X

Proyecto que desarrolla un flujo completo de clasificación binaria (Neumonía/Normal) en radiografías de tórax utilizando DenseNet-121 con transfer learning. Incluye calibración de probabilidades y análisis de explicabilidad mediante Grad-CAM. Además, integra un demo interactivo para subir imágenes y visualizar el mapa de atención, junto con un script independiente que permite validar cuantitativamente la saliencia sin modificar el cuaderno principal.

---

## Archivos principales

### chest_aid_app.py
Aplicación Gradio para realizar inferencias sobre imágenes cargadas por el usuario.  
Carga el modelo y el escalador de temperatura.  
Permite ajustar el umbral de decisión y la transparencia del Grad-CAM.  
Muestra la imagen, el score calibrado y el mapa de calor superpuesto.  
Debe de correrse todo el cuaderno de primero y luego se corre este archivo.

### chestaid.ipynb
Cuaderno de trabajo con el pipeline de entrenamiento y evaluación.  
Incluye la preparación de datos (train/val/test), el fine-tuning de DenseNet-121 y las métricas de validación y test.  
Realiza calibración de probabilidades (Temperature Scaling, Platt, Isotónica) y selección del mejor método.  
Genera Grad-CAM para ejemplos de interés.  

### saliencyextra.py
Script independiente para la validación cuantitativa de saliencia.  
Calcula curvas de sufficiency y deletion para distintos valores de k_percent.  
Genera los archivos `saliency_summary.json` (resumen de métricas y configuración) y `saliency_curves_avg.csv` (curvas promedio).  
Diseñado para ejecutarse sobre un conjunto de imágenes en disco con estructura NORMAL/PNEUMONIA.  

---

## Pesos y calibración

### chest_aid_densenet121.pth
Pesos del modelo DenseNet-121 fine-tuneado para la tarea binaria.  

### temperature_scaler.pth
Parámetro del escalado de temperatura seleccionado para calibrar las probabilidades en inferencia.  

### calibrator_meta.json
Metadatos del calibrador seleccionado (tipo de calibración y valores asociados) para ser utilizados por la aplicación.  

---

## Cómo correrlo

Primero se deben ejecutar todos los bloques del cuaderno `chestaid.ipynb`, de principio a fin, para que se entrene el modelo y se guarden los pesos junto con los parámetros de calibración.

Una vez completado ese paso, se puede correr la interfaz gráfica de **ChestAid**.  
Para iniciarla, basta con ejecutar el siguiente comando en la terminal:

```bash
python chest_aid_app.py
```
Esto cargará el modelo, aplicará la calibración y permitirá realizar inferencias visuales con Grad-CAM.

Como opción adicional, se puede ejecutar el script saliencyextra.py para validar cuantitativamente la saliencia.
La ejecución está comentada dentro del mismo archivo y corresponde al siguiente comando:

```bash
python saliencyextra.py --weights chest_aid_densenet121.pth --test_dir ./test --out_dir ./saliency_metrics --t 1.171 --n_images 64 --deletion_fill blur --sufficiency_fill mean
```
