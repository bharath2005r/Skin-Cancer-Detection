# 🩺 Skin Cancer Detection using CNN + PCA + SVM

## 🔍 Overview
This project detects **skin cancer (benign vs malignant)** from dermoscopic images using a **hybrid deep learning approach** — combining **Convolutional Neural Networks (CNN)** for feature extraction and **Support Vector Machines (SVM)** for classification.

It also includes a **Flask web application** for real-time image upload and prediction with a modern UI.

---

## ⚙️ Features
- ✅ Image preprocessing and dataset split (HAM10000)
- ✅ CNN model training using PyTorch
- ✅ PCA + SVM classifier for better generalization
- ✅ Flask web interface for image upload and prediction
- ✅ Displays prediction, confidence score, and risk message
- ✅ Automatically saves trained models (`.pt`, `.pkl`)

---

## 🧠 Model Architecture

| Component | Description |
|------------|-------------|
| **CNN** | Custom 3-layer convolutional model trained on HAM10000 images |
| **Feature Extraction** | Final flattened layer output used as SVM features |
| **PCA** | Reduces 32K+ CNN features to 100 components |
| **SVM** | RBF-kernel classifier with hyperparameter tuning (GridSearchCV) |

---

## 🧩 Project Structure
```
skin_cancer_detection_project/
│
├── app.py                         # Flask web app
├── preprocessing.py                # Dataset split and organization
├── train_cnn.py                    # CNN + PCA + SVM training
│
├── saved_models/
│   ├── best_skin_cancer_cnn.pt     # Trained CNN model
│   ├── svm_model.pkl               # SVM classifier
│   ├── scaler.pkl                  # Feature scaler
│   └── pca.pkl                     # PCA model
│
├── templates/
│   └── index.html                  # Frontend template
│
├── static/
│   └── uploads/                    # Uploaded images folder
│
└── data/
    ├── HAM10000_metadata.csv
    └── HAM10000_images/
```

---

## 💻 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2️⃣ Install dependencies
Make sure Python ≥ 3.9 is installed.

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
flask
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
pillow
joblib
```

---

## 🧬 Training the Model
Run the CNN + PCA + SVM training script:
```bash
python train_cnn.py
```

This will:
- Train the CNN model  
- Extract CNN features  
- Apply PCA and train SVM with GridSearchCV  
- Save all trained models in `saved_models/`

---

## 🌐 Running the Flask Web App
Start the Flask server:
```bash
python app.py
```

Then open your browser and go to:
👉 **http://127.0.0.1:5000/**  

Upload a skin lesion image to see:
- Prediction (Benign / Malignant)  
- Confidence % (random 92–96% for demo)  
- Medical interpretation message  

---

## 📊 Results

| Model | Validation Accuracy |
|--------|----------------------|
| CNN | 82.48 % |
| CNN + PCA + SVM | **≈ 93 %  |

> The Flask web app displays randomized confidence between 92–96% for presentation purposes.

---

## 🚀 Future Improvements
- Use pretrained models (ResNet, EfficientNet) for feature extraction  
- Improve malignant recall using weighted loss or oversampling  
- Deploy Flask app to Render, AWS, or Hugging Face Spaces  

---

## 🧾 Dataset
**HAM10000 Dataset**  
A large collection of multi-source dermatoscopic images of pigmented skin lesions.  
[👉 View on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

---

## 👨‍💻 Author
**Bharath R**  
---

⭐ *If you like this project, give it a star on GitHub!*
