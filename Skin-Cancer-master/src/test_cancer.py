import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_skin_cancer_cnn.pt')

# -----------------------------
# DEFINE SAME CNN MODEL (MATCH TRAINED VERSION)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Match your trained model: 128 * 16 * 16 → 128 → 1
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# -----------------------------
# LOAD MODEL WITH FLEXIBLE KEYS
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)

# 🔧 Rename keys if old names used
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("conv."):
        new_key = key.replace("conv.", "conv_layers.")
    elif key.startswith("fc."):
        new_key = key.replace("fc.", "fc_layers.")
    else:
        new_key = key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# -----------------------------
# TRANSFORM (must match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match your trained image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = torch.sigmoid(outputs).item()
        label = 'NOT CANCER' if predicted >= 0.5 else 'CANCER'

    print(f"🩻 Image: {os.path.basename(image_path)} → Predicted: {label.capitalize()}")
    return label

# -----------------------------
# TEST WITH YOUR OWN IMAGE
# -----------------------------
if __name__ == "__main__":
    test_image_path = input("Enter image path: ").strip().strip('"')
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print("❌ Invalid path! Please check the image file.")
