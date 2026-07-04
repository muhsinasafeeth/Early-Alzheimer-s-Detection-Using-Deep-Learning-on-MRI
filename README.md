# Early Alzheimer's Detection Using Deep Learning on MRI
### Binary Classification: NonDemented vs Very Mild Demented  
**Author: Muhsina Safeeth**

---

## 📌 Overview
This project builds a deep learning pipeline to detect early-stage Alzheimer's disease using 2D MRI slices. The focus is on distinguishing **NonDemented** from **Very Mild Demented (VMD)** cases.

---

## 🧠 Motivation
Early detection enables timely intervention. MRI contains subtle structural biomarkers such as hippocampal atrophy and ventricular enlargement. Deep learning can help identify these patterns automatically.

---

## 📂 Dataset
**Kaggle: Alzheimer’s Disease Multiclass MRI Dataset**  
Classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented  
Balanced & augmented.

### ⚠ Limitation
Dataset lacks patient IDs → risk of data leakage.  
**Solution:** Pseudo patient‑level grouping based on filename patterns.

---

## 🔧 Methods
### Phase 1–2: Data Audit & Preprocessing
- Pseudo patient grouping  
- 224×224 resizing  
- Normalization  
- Binary subset creation  
- Train/Val/Test split

### Phase 3: Baseline CNN
- 3‑layer CNN  
- Established baseline performance

### Phase 4: Transfer Learning
- **ResNet50 pretrained on ImageNet**  
- Frozen backbone → classifier head  
- Fine‑tuned last 20 layers

### Phase 5: Clinical Evaluation
Metrics:
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- F1-score  
- ROC-AUC  

### Phase 6: Explainability
Grad‑CAM visualizations highlight:
- Hippocampus  
- Medial temporal lobe  
- Ventricular regions  

### Phase 7: Robustness Check
External dataset: **OASIS MRI**  
Model generalizes beyond Kaggle.

---

## 🚀 Deployment
A Streamlit demo allows:
- MRI upload  
- Prediction  
- Grad‑CAM heatmap visualization  

---

## 📌 Future Work
- 3D CNNs for volumetric MRI  
- ADNI dataset integration  
- Multi-class dementia staging  
- Clinical validation

---

## 📜 Citation
Please cite the Kaggle dataset and OASIS dataset when using this repository.

---

## 📞 Contact
For questions or collaboration: **Muhsina Safeeth**
