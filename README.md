Hierarchical Object Detection System
This project implements a hierarchical object detection system using YOLOv8, OpenCV, and Python. The system detects objects and their associated sub-objects in a hierarchical structure, generates outputs in JSON format, and saves cropped images of detected sub-objects. It also includes functionality to benchmark inference speed (FPS) on a CPU.

Features
Hierarchical Detection:

Detects parent objects (e.g., "Person") and their associated sub-objects (e.g., "Helmet," "Tire") in video frames.
Links sub-objects to parent objects based on spatial proximity (IoU).
JSON Output:

Outputs detection results in a hierarchical JSON structure.
Image Cropping:

Crops and saves sub-object images from the video frames.
Inference Speed Benchmarking:

Measures and displays the system's FPS during video processing.
Installation
Prerequisites
Install Python 3.8 or higher.

Install the required libraries:

pip install ultralytics opencv-python-headless numpy
Download the YOLOv8 model weights:

Use the YOLOv8 Nano model for CPU optimization (yolov8n.pt).
Download from the official Ultralytics YOLOv8 repository.
Usage
1. Input Video
Place the input video in the project directory.
Update the video_path variable in the script with the path to your video file.
2. Define Object Classes
Update the parent_classes and subobject_classes lists in the script:
Replace the IDs with the appropriate class IDs from the YOLO model.
For example:
python
Copy code
parent_classes = [0]  # Person
subobject_classes = [1, 2]  # Helmet, Tire
3. Run the Script
Execute the script:
python assignment.py
The script will:

Process the video.
Save the detection results in JSON format.
Save cropped images of sub-objects.
Output
JSON File:
