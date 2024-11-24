import streamlit as st
import cv2
import tempfile
import os
from io import BytesIO
from ultralytics import YOLO
import time

# Mapping of product names to emojis
emoji_dict = {
    'lemon': '🍋',
    'carrot': '🥕',
    'potato': '🥔',
    'apple': '🍎',
    'banana': '🍌',
    'orange': '🍊',
    'broccoli': '🥦',
    'tomato': '🍅',
    'grapes': '🍇',
    'strawberry': '🍓',
    'watermelon': '🍉',
    'peach': '🍑',
    'cherry': '🍒',
    'pineapple': '🍍',
    'avocado': '🥑',
    'pear': '🍐',
    'mango': '🥭',
    'eggplant': '🍆',
    'lettuce': '🥬',
    'cucumber': '🥒',
    'zucchini': '🥒',
    'onion': '🧅',
    'garlic': '🧄',

    # Dairy and Meat Products
    'milk': '🥛',
    'cheese': '🧀',
    'butter': '🧈',
    'egg': '🥚',
    'chicken': '🍗',
    'steak': '🥩',
    'bacon': '🥓',
    'hamburger': '🍔',
    'hotdog': '🌭',
    'fish': '🐟',
    'sausage': '🌭',

    # Other items
    'bread': '🍞',
    'pizza': '🍕',
    'cake': '🍰',
    'cookie': '🍪',
    'ice cream': '🍦',
    'coffee': '☕',
    'soda': '🥤',
    'beer': '🍺',
    'wine': '🍷',
    'cocktail': '🍸',
    'champagne': '🍾',

    # Default to apple if not found
    'default': '🍏',
}


# Function to load YOLO model
def load_model(model_path: str):
    """
    Load a YOLO model from the given path.
    :param model_path: The path to the YOLO model file.
    :return: The loaded YOLO model.
    """
    model = YOLO(model_path)
    return model


# Function to process each frame using YOLO and draw bounding boxes with custom color
def process_frame(frame, model, box_color, confidence_threshold):
    """
    Process a single video frame using the YOLO model to detect objects and annotate the frame.
    :param frame: The video frame to process (in BGR format).
    :param model: The YOLO model to use for detection.
    :param box_color: The color of the bounding boxes (BGR tuple).
    :param confidence_threshold: The minimum confidence score to display an object.
    :return: The annotated frame and the detection information.
    """
    # Perform detection
    results = model(frame)

    detection_info = []  # List to store detection info for the current frame

    # Extract detected objects and draw bounding boxes
    for result in results[0].boxes.data:  # Iterate through the detected boxes
        x1, y1, x2, y2 = result[:4]
        confidence = result[4].item()
        label = int(result[5].item())

        # Filter by confidence threshold (scale the threshold to 0-1 range)
        if confidence < confidence_threshold / 100:
            continue

        # Get the class name directly from the model (assumes model has attribute `names`)
        class_name = model.names[label] if label < len(model.names) else "Unknown"

        # Find corresponding emoji for the detected class
        emoji = emoji_dict.get(class_name.lower(), emoji_dict['default'])  # Default to 🍏 if no specific match

        # Draw rectangle and label on the frame with custom box color
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, f"{emoji} {class_name} Conf: {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Add this detection to the frame's detection info (with emoji)
        detection_info.append(f"{emoji} {class_name} (Conf: {confidence:.2f})")

    # Return the annotated frame and all detections in the current frame
    return frame, detection_info



# Function to handle video file and run YOLO detection with additional controls
# Function to handle video file and run YOLO detection with additional controls
def process_video_streamlit(video_file, model_path, speed, box_color, confidence_threshold):
    # Initialize the YOLO model
    model = load_model(model_path)

    # Save the uploaded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file.close()

    # Open the video file using OpenCV
    video = cv2.VideoCapture(temp_file.name)

    # Get the FPS (frames per second) to control playback speed
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1, int(fps / speed))  # Adjust delay based on speed

    # Create a Streamlit placeholder to display frames
    stframe = st.empty()

    # Initialize the progress bar
    progress_bar = st.progress(0)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Create two columns for side-by-side layout (left for product list, right for video)
    col1, col2 = st.columns([1, 3])

    with col1:  # Column 1: Product List
        st.subheader("Detected Products 🛒")
        detected_list = st.empty()  # Placeholder for displaying the current detected product

    with col2:  # Column 2: Video Stream
        stframe = st.empty()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process each frame using the YOLO model
        frame, detection_info = process_frame(frame, model, box_color, confidence_threshold)

        # Convert BGR to RGB for Streamlit (Streamlit expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit (in the right column)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update the list of all detected products
        if detection_info:
            # Display all detections as a list
            detected_list.subheader("Detected Products 🛒")
            for info in detection_info:
                detected_list.text(info)

        # Update progress bar
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

        # Wait for the appropriate amount of time to control video speed
        time.sleep(frame_delay / fps)

    # Release video capture and delete temporary file after processing
    video.release()
    os.remove(temp_file.name)

    # Final detection list showing only the most recent product
    st.subheader("Final Detected Products 🛒")
    for product in detection_info:
        st.write(product)  # Display all final detected products
