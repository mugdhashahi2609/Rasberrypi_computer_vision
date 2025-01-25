#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include "yolo_utils.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to capture and process frames from one camera
void processCamera(int cameraId, cv::dnn::Net& net, const vector<string>& classNames, float confThreshold, float nmsThreshold) {
    // Set device paths based on cameraId
    string devicePath = (cameraId == 0) ? "/dev/video0" : "/dev/video2";  // Only video0 and video2 are available

    // Open the camera using the V4L2 backend
    VideoCapture cap(devicePath, cv::CAP_V4L2);  // Use V4L2 backend explicitly
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera " << cameraId << " at " << devicePath << endl;
        return;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Perform detection
        vector<Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;
        detectObjects(frame, net, boxes, classIds, confidences, confThreshold, nmsThreshold);

        // Draw bounding boxes on the frame
        drawDetections(frame, boxes, classIds, confidences, classNames);

        // Display the frame
        imshow("YOLOv4-tiny - Camera " + to_string(cameraId), frame);

        // Exit on ESC key
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

int main() {
    // Load YOLOv4-tiny model files
    const string classFile = "/home/mugdha/akshay/object_code/yolo4_tiny/models/coco.names";  
    const string modelConfiguration = "/home/mugdha/akshay/object_code/yolo4_tiny/models/yolov4-tiny.cfg";
    const string modelWeights = "/home/mugdha/akshay/object_code/yolo4_tiny/models/yolov4-tiny.weights";

    // Confidence and NMS Thresholds
    const float confThreshold = 0.5f;  // Confidence threshold
    const float nmsThreshold = 0.4f;   // Non-maxima suppression threshold

    // Load the YOLO model
    dnn::Net net;
    initializeYOLO(net, modelConfiguration, modelWeights);

    // Load class names
    vector<string> classNames = loadClassNames(classFile);

    // Start threads for both cameras (camera 0 and camera 2)
    thread camera1Thread(processCamera, 0, ref(net), ref(classNames), confThreshold, nmsThreshold);  // First camera (video0)
    thread camera2Thread(processCamera, 2, ref(net), ref(classNames), confThreshold, nmsThreshold);  // Second camera (video2)

    // Wait for both threads to finish
    camera1Thread.join();
    camera2Thread.join();

    return 0;
}
