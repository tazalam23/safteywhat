import cv2
import json
from ultralytics import YOLO
import os
import time


# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use YOLO Nano model for better CPU performance


# Function to perform object detection
def detect_objects(frame):
    results = model(frame)  # Perform detection
    detections = []
    for result in results[0].boxes:
        class_id = int(result.cls)  # Object class ID
        bbox = result.xyxy[0].tolist()  # Bounding box [x1, y1, x2, y2]
        confidence = float(result.conf)  # Confidence score
        detections.append({"class_id": class_id, "bbox": bbox, "confidence": confidence})
    return detections


# Function to compute Intersection Over Union (IoU) for bounding boxes
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


# Function to associate objects and sub-objects based on proximity (hierarchical relationship)
def associate_objects(detections, parent_classes, subobject_classes):
    associations = []
    parent_objects = [d for d in detections if d["class_id"] in parent_classes]
    sub_objects = [d for d in detections if d["class_id"] in subobject_classes]

    for parent in parent_objects:
        parent_bbox = parent["bbox"]
        linked_subobjects = []
        for sub in sub_objects:
            sub_bbox = sub["bbox"]
            # Associate sub-objects with the parent if IoU is greater than a threshold
            if iou(parent_bbox, sub_bbox) > 0.3:
                linked_subobjects.append(sub)
        associations.append({"parent": parent, "subobjects": linked_subobjects})
    return associations


# Function to generate JSON output for detections
def generate_json_output(associations):
    json_output = []
    for assoc in associations:
        parent = assoc["parent"]
        parent_data = {
            "object": "Parent_" + str(parent["class_id"]),
            "id": id(parent),
            "bbox": parent["bbox"],
            "subobjects": []
        }
        for sub in assoc["subobjects"]:
            sub_data = {
                "object": "Sub_" + str(sub["class_id"]),
                "id": id(sub),
                "bbox": sub["bbox"]
            }
            parent_data["subobjects"].append(sub_data)
        json_output.append(parent_data)
    return json_output


# Function to crop and save sub-object images
def crop_and_save_subobject(frame, bbox, output_path):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)


# Function to process video and generate outputs
def process_video(video_path, output_folder, parent_classes, subobject_classes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    json_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = detect_objects(frame)  # Detect objects in the frame
        associations = associate_objects(detections, parent_classes, subobject_classes)
        json_output = generate_json_output(associations)
        json_results.extend(json_output)

        # Save cropped images for sub-objects
        for assoc in associations:
            for sub in assoc["subobjects"]:
                output_path = os.path.join(
                    output_folder,
                    f"frame_{frame_count}_sub_{sub['class_id']}.jpg"
                )
                crop_and_save_subobject(frame, sub["bbox"], output_path)

    cap.release()

    # Save JSON results
    with open(os.path.join(output_folder, "results.json"), "w") as json_file:
        json.dump(json_results, json_file, indent=4)


# Function to benchmark inference speed (FPS)
def benchmark_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detect_objects(frame)
        num_frames += 1

    end_time = time.time()
    fps = num_frames / (end_time - start_time)
    cap.release()
    print(f"Inference FPS: {fps:.2f}")
    return fps


# Main function
if __name__ == "__main__":
    # Input video and output folder
    video_path = "input_video.mp4"  # Replace with your video path
    output_folder = "output"

    # Define parent and sub-object classes (replace IDs with actual class IDs)
    parent_classes = [0]  # Example: Person class ID
    subobject_classes = [1, 2]  # Example: Helmet and Tire class IDs

    # Process video
    print("Processing video...")
    process_video(video_path, output_folder, parent_classes, subobject_classes)
    print(f"Results saved in {output_folder}")

    # Benchmark inference speed
    print("Benchmarking inference speed...")
    fps = benchmark_inference(video_path)
    print(f"Achieved FPS: {fps}")
