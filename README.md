# 🧠 Early-Stage Alzheimer's Detection
Live : https://alzheimer-s-prediction-using-cnn-vuxuaaesyae32njnhvthrq.streamlit.app/
A Streamlit web app that classifies brain MRI scans as **Non-Demented** or
**Very Mild Demented** (early-stage Alzheimer's) using a custom-trained CNN,
with **Grad-CAM** visualizations to explain each prediction.

> ⚠️ **Disclaimer:** This project is for educational/research purposes only.
> It is **not** a certified medical device and must not be used for real
> clinical diagnosis.

---

## 📌 Overview

- **Input:** Brain MRI image (JPG/PNG)
- **Output:** Binary prediction — `0 = Non-Demented`, `1 = Very Mild Demented`
- **Model:** Custom CNN (3× Conv2D/MaxPooling → GlobalAveragePooling2D →
  Dense(128) → Dropout(0.5) → Dense(1, sigmoid))
- **Input size:** 224 × 224 × 3 (RGB), pixels scaled to `[0, 1]`
- **Explainability:** Grad-CAM heatmaps highlight the regions of the scan
  that most influenced the prediction

---

## 📁 Repository Structure

```
alzheimer-app/
├── app.py                     # Main Streamlit application
├── gradcam.py                 # Grad-CAM implementation
├── preprocess.py              # Image loading & preprocessing (matches training)
├── alzheimer_model_.keras     # Trained model weights
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/alzheimer-app.git
   cd alzheimer-app
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public, or private on a paid plan).
   Make sure `alzheimer_model_.keras` is committed — it's small enough
   (~470 KB) to commit directly, no Git LFS required.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select this repository and branch.
4. Set the **main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` automatically.

That's it — no secrets or external services are required for this app.

---

## 🧩 How Grad-CAM Works Here

1. The last `Conv2D` layer in the model is auto-detected
   (`gradcam.find_last_conv_layer`).
2. Gradients of the predicted class score with respect to that layer's
   feature maps are computed.
3. These gradients are global-average-pooled into per-channel importance
   weights, which are used to produce a weighted sum of the feature maps.
4. The result is ReLU'd and normalized into a `[0, 1]` heatmap, resized to
   the original image, colorized (`COLORMAP_JET`), and blended with the
   original scan to produce the overlay shown in the app.

---

## 🖥️ App Features

- 📤 MRI image upload (JPG/PNG) with error handling for invalid files
- 🎯 Prediction with class label and confidence score
- 📊 Confidence bar chart (Non-Demented vs. Very Mild Demented probability)
- 🔥 Grad-CAM heatmap + overlay, shown side-by-side with the original image
- 💾 Download button to save the Grad-CAM overlay as a PNG
- ℹ️ Sidebar with model architecture details and a medical disclaimer

---

## 📸 Screenshots

_Add screenshots of the running app here after deployment, e.g.:_

```
screenshots/
├── upload_screen.png
├── prediction_result.png
└── gradcam_overlay.png
```

---

## 🏷️ Model Training

The model was trained in `Early_Stage_Alzheimer_s_Detection_.ipynb` on a
subset (NonDemented, VeryMildDemented) of an MRI dataset from Kaggle, with
a patient-level train/val/test split to avoid data leakage. See the
notebook for full training, evaluation (accuracy/precision/recall/F1/ROC),
and Grad-CAM development details.
