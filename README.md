# 🛰️ Satellite Imagery Semantic Segmentation — U-Net on Dubai Aerial Data

> Pixel-wise land cover classification from satellite imagery using a custom U-Net — trained, evaluated, and deployed as a live web app on HuggingFace Spaces.

🔗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/coderr8/Satellite-Segmentation-Prediction)

<img width="940" height="453" alt="image" src="https://github.com/user-attachments/assets/63dc036a-2e6a-42d5-8e77-8ea9fb3a85de" />

---

## Why This Project?

Semantic segmentation of satellite imagery is a core problem in **remote sensing, urban planning, and environmental monitoring** — yet most solutions rely on massive pretrained models requiring significant compute. This project explores how far a **lightweight custom U-Net (~7.4 MB)**, trained from scratch on a small dataset (~945 patches), can go — and the results are competitive.

---

## What Was Built

An end-to-end pipeline covering data engineering, model design, training, evaluation, interpretability, and deployment — classifying every pixel in a satellite image into one of **6 land cover classes**:

| ID | Class | Color |
|----|-------|-------|
| 0 | Water | `#E2A929` |
| 1 | Land (bare) | `#8429F6` |
| 2 | Road | `#6EC1E4` |
| 3 | Building | `#3C1098` |
| 4 | Vegetation | `#FEDD3A` |
| 5 | Unlabeled | `#9B9B9B` |

---

## Dataset

**Dubai Semantic Segmentation Dataset** (Kaggle — MBRSC / Dubai Municipality)

- 8 geographic tiles × 9 image-mask pairs = **72 raw pairs**
- Patchified into **945 non-overlapping 256×256 patches**
- Split: **803 train / 142 test** (85/15)

The dataset is small by deep learning standards — making the data pipeline and loss function design especially critical.

---

## Data Pipeline — Key Decisions

| Step | What & Why |
|------|-----------|
| **Patchification** | Raw tiles are large and irregular. Cropping to 256×256 ensures uniform input and multiplies training samples |
| **MinMax Normalisation** | Per-patch scaling stabilises gradient flow — standard for satellite imagery with varying exposure |
| **RGB → Integer label** | Mask pixels carry colour-coded class information; converted to scalar labels for loss computation |
| **One-hot encoding** | Required for categorical cross-entropy based losses in multi-class segmentation |

---

## Model — Custom U-Net

Built from scratch in Keras rather than using a pretrained backbone — to demonstrate understanding of the architecture and keep the model deployable on CPU.

```
Input (256×256×3)
  → Encoder: Conv blocks at 16 → 32 → 64 → 128 → 256 filters (MaxPool between each)
  → Bottleneck: 256 filters
  → Decoder: Transposed Conv + Skip Connections (256 → 128 → 64 → 32 → 16)
  → Output: 1×1 Conv → Softmax (6 classes)
```

- **Dropout (0.2)** at every block — regularisation on a small dataset
- **Skip connections** — preserve spatial detail lost during downsampling
- **Total parameters: 1,941,190 (~7.4 MB)**

---

## Loss Function — Why Combined Loss?

A single loss wasn't enough here:

- **Dice Loss** — directly optimises overlap between predicted and ground truth masks; handles class imbalance better than cross-entropy alone
- **Categorical Focal Loss** — down-weights easy pixels, forces the model to focus on hard boundaries and rare classes
- **Equal class weights [0.1666 × 6]** — prevents dominant classes (land, water) from overwhelming minority ones (road, building)

```python
total_loss = dice_loss + (1 × focal_loss)
```

---

## Results

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | **90.62%** | **83.19%** |
| Jaccard / IoU | **0.7998** | **0.6811** |
| Loss | **0.8851** | **0.9233** |

> Trained for 70 epochs with consistent improvement and no divergence — strong generalisation for a model trained on under 1,000 patches with no data augmentation.

---

## Interpretability

Rather than treating the model as a black box, intermediate layer activations were extracted via a custom Keras sub-model to visualise **what the encoder learns** at different depths — confirming edge and texture sensitivity in early layers and semantic abstraction in deeper ones.

---

## Deployment

Exported to `.h5` and deployed on **HuggingFace Spaces** with a **Gradio Blocks** interface — users can upload any satellite/aerial image and receive a predicted segmentation mask in real time.

```python
image = Image.open("your_image.png").resize((256, 256)).convert("RGB")
prediction = model.predict(np.expand_dims(np.array(image), 0))
mask = np.argmax(prediction, axis=3)[0]
```

🔗 [Try it live](https://huggingface.co/spaces/coderr8/Satellite-Segmentation-Prediction)

