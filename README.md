# YOLOv4-Tiny Object Detection

This project demonstrates real-time object detection using the YOLOv4-tiny model with OpenCV and multi-threaded camera streaming. The application processes frames from two connected cameras and performs object detection on each stream simultaneously using OpenCV's DNN module.

## Features

- **Multi-Camera Support**: Captures and processes frames from two cameras simultaneously (`/dev/video0` and `/dev/video2`).
- **Real-Time Object Detection**: Utilizes YOLOv4-tiny for detecting objects in the captured frames.
- **Multi-Threaded Processing**: Uses threads for independent camera feeds to achieve real-time performance.
- **Easy to Extend**: The system can be easily extended to support more cameras by modifying the thread and camera ID configurations.

## Prerequisites

Ensure that the following are installed on your system:

- **OpenCV** (with the `dnn` module for deep learning support)
- **YOLOv4-Tiny Model Files**:
    - `yolov4-tiny.cfg` (model configuration)
    - `yolov4-tiny.weights` (model weights)
    - `coco.names` (class labels)

### Installation

1. **Install OpenCV**:
    - Install OpenCV with the necessary dependencies for `dnn`:
      ```bash
      sudo apt-get install libopencv-dev
      ```

2. **Download YOLOv4-Tiny Model Files**:
    - Download the model weights, configuration, and class files from the official YOLO repository or your own source.

3. **Clone the Repository**:
    - Clone the repository to your local machine:
      ```bash
      git clone https://github.com/yourusername/yolo4_tiny.git
      cd yolo4_tiny
      ```

4. **Install Required Dependencies**:
    - Install C++ dependencies:
      ```bash
      sudo apt-get install build-essential
      sudo apt-get install cmake
      ```

5. **Compile the Code**:
    - Run `make` to build the project:
      ```bash
      make
      ```

## Usage

1. **Prepare the Camera Devices**:
    - Make sure the cameras are connected to your system. You can check available video devices by running:
      ```bash
      ls -l /dev/video*
      ```

2. **Run the Object Detection**:
    - To start the object detection on both cameras, execute the following command:
      ```bash
      ./yolov4_tiny
      ```

3. **View the Detected Objects**:
    - The detected objects will be displayed in separate windows for each camera. The windows will show bounding boxes and labels on the detected objects.

4. **Exit the Program**:
    - Press the `ESC` key in any window to stop the program.

## Code Explanation

- **processCamera**: This function opens a camera stream and continuously captures frames. It performs object detection using the YOLOv4-tiny model and displays the results.
- **multi-threading**: The program uses `std::thread` to process two camera feeds in parallel.
- **YOLOv4-tiny Model**: The model is initialized with weights and configuration files. It is used to detect objects in each frame.
- **Non-Maximum Suppression (NMS)**: NMS is applied to remove duplicate detections.

### Main File
- **`main.cpp`**: This is the entry point of the program. It loads the YOLO model, sets up class names, and spawns threads for each camera feed.

## Troubleshooting

- **Camera Timeout Error**: If you encounter a `select() timeout` error, ensure that the camera is properly connected and accessible. You can test the camera using the `v4l2-ctl` tool.
- **Threading Issues**: Ensure that OpenCV is compiled with thread support if using multi-threading.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. Contributions, bug fixes, and feature requests are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
