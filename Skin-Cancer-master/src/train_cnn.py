import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib  # <-- for saving SVM, PCA, and Scaler

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚙️ Using device: {device}")

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'split_data')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

train_dir = os.path.join(DATA_DIR, 'train')
test_dir = os.path.join(DATA_DIR, 'test')

# -----------------------------
# DATA TRANSFORMS
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -----------------------------
# DATA LOADERS
# -----------------------------
train_data = datasets.ImageFolder(train_dir, transform=transform_train)
test_data = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"🖼️ Training images: {len(train_data)} | Testing images: {len(test_data)}")

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        features = x  # Features for SVM
        x = self.fc(x)
        return x, features

model = SkinCancerCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAINING LOOP
# -----------------------------
EPOCHS = 15
best_acc = 0
train_acc_list, val_acc_list = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_acc_list.append(train_accuracy)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs, _ = model(images)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_accuracy = 100 * val_correct / val_total
    val_acc_list.append(val_accuracy)

    print(f"📘 Epoch [{epoch+1}/{EPOCHS}] → Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_skin_cancer_cnn.pt"))

print(f"\n✅ CNN training complete. Best validation accuracy: {best_acc:.2f}%")
print(f"📦 CNN model saved at: {os.path.join(SAVE_DIR, 'best_skin_cancer_cnn.pt')}")

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(dataloader):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            _, feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

print("\n🔍 Extracting deep CNN features for SVM...")
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)
print(f"Feature vector shape: {X_train.shape}")

# -----------------------------
# STANDARDIZATION + PCA
# -----------------------------
print("⚙️ Standardizing and applying PCA...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA reduced feature shape: {X_train_pca.shape}")

# -----------------------------
# SVM HYPERPARAMETER TUNING
# -----------------------------
print("\n🔎 Running GridSearchCV for best SVM parameters...")
param_grid = {
    'C': [1, 10, 50],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_pca, y_train)

print(f"\n🏆 Best SVM Parameters: {grid.best_params_}")
best_svm = grid.best_estimator_

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = best_svm.predict(X_test_pca)
svm_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ CNN + PCA + SVM Accuracy: {svm_acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=train_data.classes))

# -----------------------------
# SAVE SVM, PCA, AND SCALER
# -----------------------------
joblib.dump(best_svm, os.path.join(SAVE_DIR, 'svm_model.pkl'))
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
joblib.dump(pca, os.path.join(SAVE_DIR, 'pca.pkl'))

print(f"\n💾 Models saved in: {SAVE_DIR}")
print(" - best_skin_cancer_cnn.pt  (CNN weights)")
print(" - svm_model.pkl            (SVM classifier)")
print(" - scaler.pkl               (for feature normalization)")
print(" - pca.pkl                  (for dimensionality reduction)")

# -----------------------------
# PLOT TRAINING CURVES
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.title('CNN Accuracy Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'training_accuracy_curve.png'))
plt.show()
