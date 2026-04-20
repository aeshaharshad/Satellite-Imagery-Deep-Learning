Deployed on HuggingFace : https://huggingface.co/spaces/coderr8/Satellite-Segmentation-Prediction
# 🛰️ Satellite Imagery Semantic Segmentation using U-Net

> Deep learning pipeline for pixel-wise land cover classification from aerial/satellite imagery using a custom U-Net architecture trained on the Dubai dataset.

---

## 📌 Overview

This project implements an end-to-end semantic segmentation pipeline on high-resolution satellite imagery. The model classifies each pixel into one of six land cover categories, enabling automated urban analysis, environmental monitoring, and geospatial intelligence — with a Gradio-powered inference UI for real-world deployment.

---

## 🗂️ Dataset

**Dubai Semantic Segmentation Dataset**
- **Source:** Kaggle (MBRSC / Dubai Municipality aerial imagery)
- **Structure:** 8 tiles × 9 image-mask pairs = 72 raw image pairs
- **Patch size:** 256 × 256 px (non-overlapping)
- **Total patches after patchification:** 945 image-mask pairs
- **Train / Test split:** 85% / 15% → **803 train, 142 test**

### Class Labels

| ID | Class | Color (Hex) |
|----|-------|-------------|
| 0 | Water | `#E2A929` |
| 1 | Land | `#8429F6` |
| 2 | Road | `#6EC1E4` |
| 3 | Building | `#3C1098` |
| 4 | Vegetation | `#FEDD3A` |
| 5 | Unlabeled | `#9B9B9B` |

---

## 🏗️ Model Architecture

**Custom U-Net** (built from scratch with Keras/TensorFlow)

| Component | Details |
|-----------|---------|
| Encoder depth | 5 levels (16 → 32 → 64 → 128 → 256 filters) |
| Decoder | Transposed convolutions + skip connections |
| Dropout | 0.2 at every conv block |
| Output activation | Softmax (6 classes) |
| Total parameters | **1,941,190 (~7.4 MB)** |
| Input shape | `(256, 256, 3)` |

---

## ⚙️ Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Dice Loss + Categorical Focal Loss (equal class weights) |
| Class weights | [0.1666 × 6] — balanced |
| Batch size | 16 |
| Epochs | 70 |
| Framework | TensorFlow / Keras + `segmentation-models` |

---

## 📊 Results

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | **90.62%** | **83.19%** |
| Jaccard Coefficient (IoU) | **0.7998** | **0.6811** |
| Loss (Dice + Focal) | **0.8851** | **0.9233** |

> Results recorded at **Epoch 70**. The model shows consistent improvement across all 70 epochs with no sign of divergence, and generalises well given the small dataset size (~945 patches).

---

## 🔬 Techniques Used

- **Patchification** — large tiles cropped to 256×256 non-overlapping patches using `patchify`
- **MinMax Normalisation** — per-patch scaling of satellite image values
- **RGB → Label encoding** — mask pixel colours mapped to integer class IDs
- **One-hot encoding** — `to_categorical` for multi-class segmentation output
- **Combined Loss** — Dice Loss addresses class imbalance; Focal Loss penalises hard misclassifications
- **Layer activation extraction** — intermediate feature maps visualised via custom sub-model (`conv2d_17`)
- **Gradio UI** — browser-based inference app for custom image upload and mask prediction

---

## 🚀 Inference

The model accepts any RGB satellite/aerial image, resizes it to 256×256, and outputs a predicted segmentation mask. A Gradio Blocks interface provides an interactive web UI.

```python
image = Image.open("your_image.png").resize((256, 256)).convert("RGB")
image = np.expand_dims(np.array(image), 0)
prediction = model.predict(image)
predicted_mask = np.argmax(prediction, axis=3)[0]
```

---

## 📁 Project Structure

```
├── Satellite_Imagery_DeepLearning.ipynb   # Data prep, training, evaluation
├── Satellite_segmentation_Prediction.ipynb # Model loading, inference, Gradio UI
├── satellite_segmentation_full.h5          # Saved model weights
└── README.md
```

---

## 🛠️ Dependencies

```
tensorflow / keras
segmentation-models
patchify
opencv-python
scikit-learn
gradio
matplotlib
numpy, pillow
```
