import streamlit as st
import cv2
from PIL import Image



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

        try:
            # Ensure cropped_image is not empty or None before processing
            if cropped_image is None or cropped_image.size == 0:
                raise ValueError(f"Empty image for label: {label}")

            # Resize the cropped image to 100x100 for better quality display
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image_resized = cv2.resize(cropped_image_rgb, (100, 100))  # Larger thumbnail size

            # Create a PIL Image from the resized NumPy array to display it in Streamlit
            pil_image = Image.fromarray(cropped_image_resized)

            # Add the image and label with confidence to the display list
            images_to_display.append((pil_image, label, f"Confidence: {confidence:.2f}"))

        except Exception as e:
            # In case of an error (e.g., cvtColor, resize, etc.), log the error and continue
            continue

    # Create a Streamlit layout to display up to 5 images per row
    cols = st.columns(5)  # 5 columns per row

    for idx, (img, label, conf) in enumerate(images_to_display):
        col_idx = idx % 5  # To display images in the current row

        with cols[col_idx]:
            # Adding some styling for the caption
            st.markdown(f"### {label}")
            # Display the image with custom styling
            st.image(img, caption=f"{label}: {confidence:.2f}", use_container_width=True)

    # Optional: Add a stylish separator to make the images look more organized
    st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)