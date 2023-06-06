import cv2
import time
from ultralytics import YOLO
import argparse
import yaml
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run empty detection on a video')
    parser.add_argument('--data', required=True, help='Path to YAML config file')
    parser.add_argument('--source', required=True, help='Path to the input video')
    parser.add_argument('--output', required=True, help='Path to save the output video')
    parser.add_argument('--weights', required=True, help='Path to checkpoint file')
    return parser.parse_args()

def apply_transparent_red(frame, xmin, ymin, xmax, ymax):
    overlay = frame.copy()
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)  # Draw the transparent red area
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)  # Draw the white rectangle border
    alpha = 0.5  # Transparency factor (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

args = parse_args()

# Load the model
model = YOLO(args.weights)

# Create output directory if it does not exist
os.makedirs(args.output, exist_ok=True)

# Load dataset parameters from YAML config file
with open(args.data, 'r') as f:
    config = yaml.safe_load(f)
classes = config['names']

# Initialize the empty detection parameters
empty_threshold = 0.6  # Confidence threshold for empty detection

# Start the video capture
cap = cv2.VideoCapture(args.source)

# Get the current time
start_time = time.time()
frame_count = 0

# Get the video frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = cv2.VideoWriter(os.path.join(args.output, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize empty detection flag and detected empty areas
empty_detected = False
empty_areas = []

while True:
    # Capture the frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    # Get the detections as a list of dictionaries
    detections = results.boxes.data.tolist()

    # Reset empty detection flag and detected empty areas
    empty_detected = False
    empty_areas = []

    # Iterate over the detections
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_idx = detection

        if confidence > empty_threshold and (classes[int(class_idx)] == 'empty' or classes[int(class_idx)] == 'emptys'):
            empty_detected = True
            empty_areas.append((int(xmin), int(ymin), int(xmax), int(ymax)))

    # If empty is detected, apply transparent red overlay to the areas covered by the bounding boxes
    if empty_detected:
        for area in empty_areas:
            xmin, ymin, xmax, ymax = area
            frame = apply_transparent_red(frame, xmin, ymin, xmax, ymax)

    # Display the FPS
    end_time = time.time()
    fps = int(frame_count / (end_time - start_time))
    cv2.putText(frame, "FPS: {}".format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Model: Yolov8x", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 138, 139), 3)

    # Write the frame to the output video
    out_video.write(frame)

    frame_count += 1

# Release the video capture and output video objects
cap.release()
out_video.release()
           