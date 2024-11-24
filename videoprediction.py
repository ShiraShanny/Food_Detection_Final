import cv2
import numpy as np
from collections import defaultdict
from fastai.vision.all import load_learner, PILImage
import streamlit as st
import tempfile
from io import BytesIO

# Class to handle video reading
class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Frame index out of range")
        return frame

    def get_avg_fps(self):
        return self.fps

    def release(self):
        self.cap.release()

# Function to load YOLO model
def load_yolo_model(weights_path="yolo3/yolov3.weights", cfg_path="yolo3/yolov3.cfg", names_path="yolo3/coco.names"):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes

# Function to process video with adjustable parameters
def process_video(video_path, learn, output_video_path, frame_skip=20, input_size=320, confidence_threshold=0.4, nms_threshold=0.4):
    vr = VideoReader(video_path)  # Initialize video reader
    fps = vr.get_avg_fps()  # Get frames per second
    frame_predictions = defaultdict(set)  # To store frame predictions
    detected_frames = []  # Frames where ingredients are detected

    # Load YOLO model
    net, output_layers, classes = load_yolo_model()

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(vr.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vr.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Iterate through video frames with adjustable frame skipping
    for frame_idx in range(len(vr)):
        if frame_idx % frame_skip != 0:
            continue

        frame = vr[frame_idx]  # Read frame

        # FastAI model prediction on frame
        img = PILImage.create(frame)
        product, _, probs = learn.predict(img)

        # Prepare frame for YOLO detection
        height, width, _ = frame.shape
        resized_frame = cv2.resize(frame, (input_size, input_size))
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        # Analyze YOLO output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:  # Only consider high confidence detections
                    center_x = int(detection[0] * input_size)
                    center_y = int(detection[1] * input_size)
                    w = int(detection[2] * input_size)
                    h = int(detection[3] * input_size)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Draw bounding box and label on frame
                    label = str(classes[class_id])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected_frames.append(frame)  # Add frame to list if ingredient detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)  # Non-maxima suppression


        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()  # Flatten the array if it's not empty
            for i in indexes:
                label = str(classes[class_ids[i]])
                frame_predictions[label].add(frame_idx // fps)

        out.write(frame)  # Write frame to output video

    vr.release()
    out.release()

    return frame_predictions, detected_frames

# Helper function to format the time range of detected ingredients
def format_time_list(time_set):
    time_ranges = sorted(time_set)
    ranges = []
    start_time = time_ranges[0]

    for i in range(1, len(time_ranges)):
        if time_ranges[i] != time_ranges[i - 1] + 1:
            end_time = time_ranges[i - 1]
            ranges.append((start_time, end_time))
            start_time = time_ranges[i]

    ranges.append((start_time, time_ranges[-1]))
    return ', '.join(f"{s}-{e} seconds" if s != e else f"{s} seconds" for s, e in ranges)

# Function to calculate performance metrics (precision, recall, F1 score)
def calculate_metrics(detected, actual):
    true_positives = len(detected & actual)
    false_positives = len(detected - actual)
    false_negatives = len(actual - detected)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

# Simple function to generate recipe from detected ingredients
def generate_recipe(ingredients):
    if ingredients:
        recipe = f"Recipe using: {', '.join(set(ingredients))}"
    else:
        recipe = "No ingredients identified."
    return recipe


def detect_all_products(video_path, learn, output_folder="output_images"):
    vr = VideoReader(video_path)  # Initialize video reader
    fps = vr.get_avg_fps()  # Get frames per second
    detected_images = []  # To store frames with detected products
    cropped_detections = []  # To store cropped detected portions along with the label

    frame_skip = 20  # Skip frames for faster processing
    input_size = 320  # YOLO input size

    # Load YOLO model
    net, output_layers, classes = load_yolo_model()

    # Iterate through video frames
    for frame_idx in range(len(vr)):
        if frame_idx % frame_skip != 0:
            continue

        frame = vr[frame_idx]  # Read frame

        # FastAI model prediction on frame
        img = PILImage.create(frame)
        product, _, probs = learn.predict(img)

        # Get original frame dimensions (height and width)
        height, width, _ = frame.shape

        # Prepare frame for YOLO detection
        resized_frame = cv2.resize(frame, (input_size, input_size))
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        # Analyze YOLO output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:  # Only consider high confidence detections
                    center_x = int(detection[0] * input_size)
                    center_y = int(detection[1] * input_size)
                    w = int(detection[2] * input_size)
                    h = int(detection[3] * input_size)

                    # Convert to top-left coordinates for the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)



        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()  # Flatten the array if it's not empty

            detected_labels = []
            for i in indexes:  # Process each detection
                x, y, w, h = boxes[i]

                # Scale back the bounding box coordinates to the original frame size
                x = int(x * (width / input_size))
                y = int(y * (height / input_size))
                w = int(w * (width / input_size))
                h = int(h * (height / input_size))

                label = classes[class_ids[i]]
                confidence = confidences[i]
                detected_labels.append(f"{label} - {confidence * 100:.2f}%")

                # Draw the corrected bounding box on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the detected region from the frame and store it with the label
                cropped_region = frame[y:y+h, x:x+w]
                cropped_detections.append({"image": cropped_region, "label": label ,"confidence": confidence})

            if detected_labels:
                detected_images.append({
                    "frame": frame,
                    "label": ", ".join(detected_labels),
                    "confidence": max(confidences)
                })


        else:
            print(f"No detections in frame {frame_idx}")

    vr.release()

    # Return cropped detections with their labels
    return cropped_detections, detected_images


def detect_and_highlight_elements(video_path, learn, frame_skip=20, input_size=320,
                                  confidence_threshold=0.5, nms_threshold=0.4):
    # Open video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return

    # Get FPS and frame dimensions from the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary file to store the output video in memory
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')

    # Setup the video writer to save the output video with the same resolution and fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID or MJPG codec for Streamlit compatibility
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (frame_width, frame_height))

    # Load YOLO model
    net, output_layers, classes = load_yolo_model()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break when video ends

        # Process every frame_skip-th frame
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # FastAI model prediction
        img = PILImage.create(frame)
        product, _, probs = learn.predict(img)

        # Prepare frame for YOLO detection
        resized_frame = cv2.resize(frame, (input_size, input_size))  # Resize for YOLO
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        # Process YOLO outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    center_x = int(detection[0] * input_size)
                    center_y = int(detection[1] * input_size)
                    w = int(detection[2] * input_size)
                    h = int(detection[3] * input_size)

                    # Convert to top-left coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        # Create a grey background with transparency (50% opacity)
        grey_frame = np.ones((frame_height, frame_width, 4), dtype=np.uint8) * 128  # (RGBA, grey background)

        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()

            for i in indexes:
                x, y, w, h = boxes[i]

                # Scale bounding box back to original frame size
                x = int(x * (frame_width / input_size))
                y = int(y * (frame_height / input_size))
                w = int(w * (frame_width / input_size))
                h = int(h * (frame_height / input_size))

                label = classes[class_ids[i]]
                confidence = confidences[i]

                # Draw the detected object on the original frame
                if product == label:  # Check if FastAI detected the product
                    color = (0, 0, 255)  # Red for detected product
                else:
                    color = (0, 0, 0)  # Black for non-detected product

                # Mask the area inside the bounding box to preserve the original frame content
                grey_frame[y:y + h, x:x + w, :3] = frame[y:y + h, x:x + w]  # Copy only the color channels (R, G, B)
                grey_frame[y:y + h, x:x + w, 3] = 255  # Set alpha channel to 255 (fully opaque for detected products)

                # Draw bounding box on grey background (inside detected product region)
                cv2.rectangle(grey_frame, (x, y), (x + w, y + h), color, 2)  # Drawing with the selected color
                cv2.putText(grey_frame, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

        # Blend original frame and grey background with transparency
        # Here alpha is 0.5 for the semi-transparent grey background
        alpha = 0.5
        grey_frame_rgb = grey_frame[:, :, :3]  # Remove the alpha channel for blending
        blended_frame = cv2.addWeighted(grey_frame_rgb, alpha, frame, 1 - alpha, 0)

        # Write the processed frame to the output video
        out.write(blended_frame)
        frame_idx += 1

    # Release video resources
    cap.release()
    out.release()

    # Open the saved video file and read it into memory as bytes
    with open(temp_output.name, 'rb') as f:
        video_data = f.read()

    # Provide download button for the processed video
    st.download_button(
        label="Download Processed Video",
        data=video_data,
        file_name="processed_video.avi",
        mime="video/avi"
    )

    # Display the processed video in Streamlit
    st.video(video_data)

    # Clean up the temporary file
    temp_output.close()



