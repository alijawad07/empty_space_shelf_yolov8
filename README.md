# Empty Space in a shelf or an aisle Detection with YOLOv8

Empty Space Detection with YOLOv8 is a computer vision project that aims to detect falls using the YOLOv8 object detection model. This project provides a real-time Empty Space detection solution by analyzing video streams.

## Features

- Utilizes the YOLOv8 object detection model for accurate fall detection
- Real-time detection and immediate alert using visual cues
- With accurate identification of vacant areas, businesses can enhance their restocking processes, improve customer experience, and maximize shelf utilization.
- Built with efficiency and ease-of-use in mind

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8

## Getting Started

1. Clone the repository:

```
https://github.com/alijawad07/empty_space_shelf_yolov8
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Update the configuration file with the appropriate paths and parameters.

4. Run the empty_shelf script:
```
python3 empty_shelf.py --data --source --output --weights
```
- --data => .yaml file with dataset and class details

- --source => Path to directory containing video

- --output => Path to save the detection results

- --weights => Path to yolov8 weights file


## Acknowledgments

- Thanks to Roboflow for providing the comprehensive fall detection dataset used in training the YOLOv8 model.
- Special appreciation to Ultralytics for developing the YOLOv8 model and its integration with the project.

## References

- [YOLOv8](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com/)
