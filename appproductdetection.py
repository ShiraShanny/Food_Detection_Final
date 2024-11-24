import streamlit as st
from fastai.vision.all import load_learner, PILImage
import recipe_generator
import videoprediction
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from pathlib import Path
from PIL import Image
import warnings
import customyolo
from geminiapi import detect_product_names
from yoloimages import display_detected_frames, display_best_detected_image
from duckduckgo import search_images, display_image_from_url, sanitize_filename
from torchvision.models import ResNet50_Weights
from ultralytics import YOLO
from io import BytesIO


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set the path where images will be saved
path = Path('foodproduct')
path.mkdir(exist_ok=True)



# Custom CNN model definition
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)


# Load the model
model_filename = 'models/fruit_cnn_model.pkl'
learn = None
if Path(model_filename).exists():
    learn = load_learner(model_filename)
else:
    st.error(f"Model file '{model_filename}' not found.")


# Function for ingredient detection from image
def detect_ingredients_from_image(uploaded_image):
    img = PILImage.create(uploaded_image)
    product, _, probs = learn.predict(img)
    return product, probs


# Function to list products above a probability threshold
def list_all_products(probs, threshold=0.7):
    return [(learn.dls.vocab[i], probs[i].item()) for i in range(len(probs)) if probs[i] > threshold]


# Define class names
CLASS_NAMES = [
    'Blueberries', 'Broccoli', 'Pasta', 'Chicken Breast', 'Fillet Salmon', 'Grounded Beef',
    'Avocado', 'Banana', 'Carrot', 'Mushrooms', 'Cucumber', 'Garlic', 'Lemon', 'Orange',
    'Pineapple', 'Apple', 'Strawberries', 'Sweet Potato', 'Tomatoe', 'Onion', 'Bell Pepper',
    'Potato', 'Lettuce', 'Cheese', 'Eggs'
]


# Load pre-trained model for transfer learning
class ImagenetTransferModeling(nn.Module):
    def __init__(self, num_classes=25):
        super(ImagenetTransferModeling, self).__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]  # Remove fully connected layer
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x


# Load the trained model from a .pth file
model_path = "models/model_and_transform.pth"
model2 = None
transform = None
if Path(model_path).exists():
    model2 = ImagenetTransferModeling(num_classes=25)

    # Load the checkpoint into the model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()

    transform_params = checkpoint.get("transform_params", {"resize": (224, 224)})
    transform = transforms.Compose([
        transforms.Resize(transform_params["resize"]),
        transforms.ToTensor(),
    ])
else:
    st.error("Model file not found. Please check the file path.")


# Prediction function for fridge sections
def classify_image_in_parts(image_path, transform, num_shelves):
    model2.eval()
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    section_height = height // num_shelves

    sections_predictions = []
    for i in range(num_shelves):
        top = i * section_height
        bottom = top + section_height if i < num_shelves - 1 else height
        section = image.crop((0, top, width, bottom))
        transformed_section = transform(section).unsqueeze(0)

        with torch.no_grad():
            logits = model2(transformed_section)
            probabilities = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)

        section_predictions = [(idx, prob) for idx, prob in
                               zip(top_indices.squeeze().tolist(), top_probs.squeeze().tolist())]
        sections_predictions.append({"section": i + 1, "predictions": section_predictions})

    return sections_predictions


# Streamlit UI
st.title("🍽️ Recipe Generator with Fridge Section Analysis")

st.sidebar.header("Choose Detection Method")
option = st.sidebar.selectbox("Select a method:", ["Image Detection", "Image Detection with API", "Video Detection","Custom Video Detection"])

if option == "Image Detection":
    st.header("📸 Image Upload for Fridge Analysis")
    uploaded_image = st.file_uploader("Upload a fridge image:", type=["jpg", "jpeg", "png"])

    if "detected_products" not in st.session_state:
        st.session_state.detected_products = set()

    if "ingredients_list" not in st.session_state:
        st.session_state.ingredients_list = []

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        num_shelves = st.number_input("How many shelves are in the fridge?", min_value=1, value=3)
        num_items = st.number_input("How many items on each shelf?", min_value=1, value=2)

        if st.button("🔍 Detect Ingredients"):
            predictions = classify_image_in_parts(uploaded_image, transform, num_shelves)
            detected_products = set(
                CLASS_NAMES[idx] for pred in predictions for idx, _ in pred["predictions"]
            )
            st.session_state.detected_products = detected_products

            for prediction in predictions:
                st.subheader(f"Shelf {prediction['section']} Predictions")
                for idx, prob in prediction["predictions"]:
                    st.write(f"{CLASS_NAMES[idx]}: {100 * prob:.2f}%")

            detected_products_text = ", ".join(sorted(st.session_state.detected_products))
            st.text_area("Detected Ingredients:", detected_products_text, height=100, disabled=True)

            ingredients_input = st.text_area("✏️ Edit Detected Ingredients (optional):", detected_products_text)
            if ingredients_input != detected_products_text:
                st.session_state.ingredients_list = [ingredient.strip() for ingredient in ingredients_input.split(",")
                                                     if ingredient.strip()]
            else:
                st.session_state.ingredients_list = list(st.session_state.detected_products)

        if st.session_state.ingredients_list:
            if st.button("🔍 Generate Recipe"):
                try:
                    recipe = recipe_generator.generate_recipe(st.session_state.ingredients_list)

                    if recipe:
                        st.subheader("📜 Generated Recipe")
                        st.write(f"**Title:** {recipe.title} 🥘")

                        search_term = recipe.title
                        image_urls = search_images(search_term, max_images=1)

                        if image_urls:
                            display_image_from_url(image_urls[0])
                        else:
                            st.write("No image found for the recipe.")

                        st.write(f"**Servings:** {recipe.servings} 🥣")
                        st.write(f"**Preparation Time:** {recipe.prep_time} ⏱️")
                        st.write(f"**Cooking Time:** {recipe.cook_time} ⏲️")
                        st.write(f"**Total Time:** {recipe.total_time} ⏳")
                        st.write(f"**Difficulty:** {recipe.difficulty} 🏆")
                        st.write(f"**Cuisine:** {recipe.cuisine} 🌍")
                        st.write(f"**Category:** {recipe.category} 📂")

                        st.subheader("🛒 Ingredients")
                        for ing in recipe.ingredients:
                            st.write(f"- {ing.amount} {ing.unit} {ing.name} 🥄")

                        st.subheader("🔪 Instructions")
                        for index, instruction in enumerate(recipe.instructions):
                            st.write(f"{index + 1}. {instruction}")
                    else:
                        st.write("Recipe generation failed.")
                except Exception as e:
                    st.error(f"Error generating recipe: {e}")

elif option == "Custom Video Detection":
    st.title("Professional YOLO Video Detection App")

    st.header("🎥 Upload Video for Custom Detection")
    uploaded_video = st.file_uploader("Upload a video file:", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.sidebar.text("Video uploaded successfully.")

        # Add options to control the video processing speed, bounding box color, and confidence threshold
        speed = st.sidebar.slider("Select Video Speed", min_value=0.1, max_value=100.0, value=20.0, step=0.5)
        box_color_choice = st.sidebar.selectbox("Select Bounding Box Color", ["Green", "Red", "Blue", "Yellow"])
        confidence_threshold = st.sidebar.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5,
                                                 step=0.05)

        # Map box color to BGR format for OpenCV
        box_colors = {
            "Green": (0, 255, 0),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255)
        }
        box_color = box_colors[box_color_choice]

        # Define YOLO model path (adjust path to your setup)
        model_path = "customyolo/best.pt"  # Path to your custom YOLO model

        # Display spinner while the video is being processed
        with st.spinner("Processing the video..."):
            # Process the video with YOLO detection
            customyolo.process_video_streamlit(uploaded_video, model_path, speed, box_color, confidence_threshold)

        st.success("Video processing completed!")