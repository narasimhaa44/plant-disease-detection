import torch
from torchvision import transforms, models, datasets
from PIL import Image
from disease_advisor import DiseaseAdvisor
import torch.nn as nn
from weather_utils import get_weather
import os

# === WeatherAPI key (local use only) ===
WEATHER_API_KEY = "88bfec56811240ffa1c143935251211"

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Model setup ===
num_classes = 38  # match your dataset
model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# Replace final FC layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# Load model weights
model.load_state_dict(
    torch.load(r"notebooks\best_cnn_model_copy_finetuned.pth", map_location=device)
)
model = model.to(device)
model.eval()
print("Model loaded successfully with correct class count.")

# === Load disease knowledge base ===
advisor = DiseaseAdvisor(r"knowledge\disease_knowledge_base.json")

# === Image transform (same as validation) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Prediction function with Weather Integration ===
def predict_with_advice(image_path, city_name):
    print(f"\nAnalyzing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    # Map label
    train_data = datasets.ImageFolder(root=r"data/train")
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
    predicted_label = idx_to_class[pred.item()]

    print(f"Predicted disease: {predicted_label}")

    # Disease info
    info = advisor.get_info(predicted_label)
    if "error" in info:
        print(info["error"])
        if "suggestion" in info:
            print(info["suggestion"])
    else:
        print("=" * 70)
        print(f"Disease: {info['disease']}")
        print(f"Description: {info['description']}")
        print("Recommendations:")
        for i, rec in enumerate(info["recommendations"], 1):
            print(f"  {i}. {rec}")
        print("=" * 70)

    # Fetch current weather
    print("\nFetching current weather...")
    weather = get_weather(city_name, WEATHER_API_KEY)

    if "error" in weather:
        print(weather["error"])
    else:
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Conditions: {weather['conditions']}")
        print(f"Rainfall (last hour): {weather['rain_1h_mm']} mm")

        # Conditional environmental advice
        if weather["humidity"] > 80:
            print("High humidity — fungal diseases may worsen, apply fungicide preventively.")
        elif weather["temperature"] > 35:
            print("High temperature — ensure plants are well-watered and shaded.")
        elif weather["rain_1h_mm"] > 0:
            print("Recent rainfall — check for leaf wetness and soil drainage.")

# === Example usage ===
if __name__ == "__main__":
    sample_path = r"data\test\TomatoEarlyBlight1.JPG"
    city_name = input("Enter your city name: ").strip() or "Bangalore"
    predict_with_advice(sample_path, city_name)
