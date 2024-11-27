import streamlit as st
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import recipe_generator
import torch
import torch.nn as nn  # Import the neural network module
import videoprediction
import cv2
from PIL import Image
import time


def display_detected_frames(detected_images):
    for idx, detection in enumerate(detected_images):
        st.image(detection["frame"], caption=f"Frame {idx + 1} - {detection['label']}, Confidence: {detection['confidence']:.2f}")


def display_best_detected_image(cropped_detections):
    best_detections = {}

    # Iterate over all cropped detections to find the best detection for each label
    for detection in cropped_detections:
        label = detection["label"]
        confidence = detection["confidence"]

        # If the label is not in best_detections or the current detection has higher confidence
        if label not in best_detections or confidence > best_detections[label]["confidence"]:
            best_detections[label] = detection

    # Create a list of images for each label
    images_to_display = []

    for label, detection in best_detections.items():
        cropped_image = detection["image"]
        confidence = detection["confidence"]

        # Resize the cropped image to 100x100 for better quality display
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_resized = cv2.resize(cropped_image_rgb, (100, 100))  # Larger thumbnail size

        # Create a PIL Image from the resized NumPy array to display it in Streamlit
        pil_image = Image.fromarray(cropped_image_resized)

        # Add the image and label with confidence to the display list
        images_to_display.append((pil_image, label, f"Confidence: {confidence:.2f}"))

    # Create a Streamlit layout to display up to 5 images per row
    cols = st.columns(5)  # 5 columns per row

    for idx, (img, label, conf) in enumerate(images_to_display):
        col_idx = idx % 5  # To display images in the current row

        with cols[col_idx]:
            # Adding some styling for the caption
            st.markdown(f"### {label}")
            # Display the image with custom styling
            st.image(img, caption="", use_column_width=True)

    # Optional: Add a stylish separator to make the images look more organized
    st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)


# Custom CNN model definition
class CustomCNN(nn.Module):
    def __init__(self):
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
        self.fc2 = nn.Linear(256, len(dls.vocab))

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
if not Path(model_filename).exists():
    st.error(f"Model file '{model_filename}' not found.")
else:
    learn = load_learner(model_filename)

# Function for ingredient detection from image
# Function for ingredient detection from image
def detect_ingredients_from_image(uploaded_image):
    img = PILImage.create(uploaded_image)
    product, _, probs = learn.predict(img)
    return product, probs

# Function to list products above a probability threshold
def list_all_products(probs, threshold=0.7):
    return [(learn.dls.vocab[i], probs[i].item()) for i in range(len(probs)) if probs[i] > threshold]

# UI for Recipe Generator
st.title("🍽️ Recipe Generator")

# Sidebar for choosing detection method
st.sidebar.header("Choose Detection Method")
option = st.sidebar.selectbox("Select a method:", ["Image Detection", "Video Detection"])

if option == "Image Detection":
    # Image upload section
    st.header("📸 Image Upload for Ingredient Detection")
    uploaded_image = st.file_uploader("Upload an image of ingredients:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        product, probs = detect_ingredients_from_image(uploaded_image)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Process identified products
        products_list = list_all_products(probs)
        detected_products = set(product for product, prob in products_list)

        if detected_products:
            detected_products_text = ", ".join(detected_products)
            st.success("✨ Ingredients detected from the image!")
            st.text_area("Detected Ingredients:", detected_products_text, height=100, disabled=True)

            ingredients_input = st.text_area("✏️ Edit Detected Ingredients (optional):", detected_products_text)

            if st.button("🔍 Generate Recipe"):
                ingredients_list = [ingredient.strip() for ingredient in ingredients_input.split(",") if ingredient.strip()]
                if ingredients_list:
                    try:
                        recipe = recipe_generator.generate_recipe(ingredients_list)

                        if recipe:
                            st.subheader("📜 Generated Recipe")
                            st.write(f"**Title:** {recipe.title} 🥘")
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
                            for index, instruction in enumerate(recipe.instructions, start=1):
                                st.write(f"{index}. {instruction}")

                            st.subheader("🍏 Nutrition Information")
                            st.json(recipe.nutrition)

                            # Added and removed products
                            st.subheader("🛍️ Added Products")
                            added_products_text = ", ".join(recipe.added_products) if recipe.added_products else "No added products found."
                            st.text_area("Added Products", value=added_products_text, height=100)

                            st.subheader("❌ Removed Products")
                            removed_products_text = ", ".join(recipe.removed_products) if recipe.removed_products else "No removed products found."
                            st.text_area("Removed Products", value=removed_products_text, height=100)

                            st.subheader("🔍 Detected Products")
                            detected_products_text = ", ".join(recipe.detected_products) if recipe.detected_products else "No detected products found."
                            st.text_area("Detected Products", value=detected_products_text, height=100)

                        else:
                            st.warning("⚠️ Recipe generation failed. Please try again.")
                    except Exception as e:
                        st.error(f"❗ An error occurred: {str(e)}")
                else:
                    st.warning("⚠️ Please edit the detected ingredients before generating a recipe.")
        else:
            st.warning("⚠️ No ingredients detected above the threshold. Please try another image.")

elif option == "Video Detection":
    # Video upload section
    st.header("🎥 Video Upload for Ingredient Detection")
    uploaded_video = st.file_uploader("Upload a video of ingredients:", type=["mp4", "mov"])

    if uploaded_video:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Show the uploaded video
        st.video("temp_video.mp4")
        st.success("✨ Video uploaded successfully!")

        # Allow the user to adjust parameters using Streamlit widgets
        st.sidebar.subheader("🛠️ Adjust Detection Settings")

        # Parameter Inputs
        frame_skip = st.sidebar.slider("Frame Skip (frames)", min_value=1, max_value=50, value=20, step=1)
        input_size = st.sidebar.slider("Input Size (pixels)", min_value=128, max_value=640, value=320, step=32)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        nms_threshold = st.sidebar.slider("NMS Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

        # Run predictions on the video with adjusted parameters
        video_predictions, detected_frames = videoprediction.process_video(
            "temp_video.mp4", learn, "output_video.mp4",
            frame_skip=frame_skip,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        if video_predictions:
            st.markdown("### 🎉 Ingredients Detected! ✨")
            st.markdown("Here are the ingredients detected in the video:")

            detected_ingredients = []
            for product, time_set in video_predictions.items():
                formatted_times = videoprediction.format_time_list(time_set)
                st.markdown(f"**🍽️ {product}** - ⏰ Seconds: {formatted_times}")
                detected_ingredients.append(product)

            st.markdown("---")  # Add a horizontal line for separation

            # Add radio button to choose between generating a recipe or showing image detection
            choice = st.radio(
                "Choose an action:",
                ("None", "Generate Recipe", "Show Full Image Detection ", "Show Cropped Image Detection", "Show Highlighted Video Frames")
            )

            if choice == "Generate Recipe":
                # Generate recipe based on detected ingredients
                if detected_ingredients:
                    try:
                        recipe = recipe_generator.generate_recipe(detected_ingredients)

                        if recipe:
                            st.subheader("📜 Generated Recipe")
                            st.write(f"**Title:** {recipe.title} 🥘")
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
                            for index, instruction in enumerate(recipe.instructions, start=1):
                                st.write(f"{index}. {instruction}")

                    except Exception as e:
                        st.error(f"❗ An error occurred: {str(e)}")
                else:
                    st.warning("⚠️ No ingredients detected to generate a recipe.")

            elif choice == "Show Image Detection":
                # Detect products in the video and display the detected frames
                cropped_detections, detected_images = videoprediction.detect_all_products("temp_video.mp4", learn)
                display_detected_frames(detected_images)
            elif choice == "Show Cropped Detection":
                # Detect products in the video and display the detected frames
                cropped_detections, detected_images = videoprediction.detect_all_products("temp_video.mp4", learn)
                display_best_detected_image(cropped_detections)
            elif choice == "Show Highlighted Video Frames":
                # Detect and highlight elements in video frames
                highlighted_video = videoprediction.detect_and_highlight_elements(
     "temp_video.mp4", learn,
                frame_skip=5,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold   )

        else:
            st.warning("⚠️ No ingredients detected in the video. Please try another video.")
            st.markdown("🔍 **Tip:** Ensure the video is clear and the ingredients are visible!")
