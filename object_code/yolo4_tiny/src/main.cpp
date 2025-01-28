#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include "yolo_utils.hpp"
#include <mutex>

using namespace cv;
using namespace cv::dnn;
using namespace std;

mutex frameMutex;  // Mutex to synchronize frame access

// Function to capture and process frames from one camera
void processCamera(int cameraId, cv::dnn::Net& net, const vector<string>& classNames, float confThreshold, float nmsThreshold, Mat &frameResult, bool &newFrame) {
    string devicePath = (cameraId == 0) ? "/dev/video0" : "/dev/video2";  // Only video0 and video2 are available

    VideoCapture cap(devicePath, cv::CAP_V4L2);  // Use V4L2 backend explicitly
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera " << cameraId << " at " << devicePath << endl;
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);  // Set resolution
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240); // Set resolution
    cap.set(cv::CAP_PROP_FPS, 15);           // Set frame rate

    Mat frame;
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            cerr << "Error: Captured empty frame from Camera " << cameraId << endl;
            continue;
        }

        // Perform object detection
        vector<Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;
        detectObjects(frame, net, boxes, classIds, confidences, confThreshold, nmsThreshold);

        // Draw bounding boxes on the frame
        drawDetections(frame, boxes, classIds, confidences, classNames);

        // Synchronize frame access using mutex
        {
            lock_guard<mutex> lock(frameMutex);
            frameResult = frame.clone();
            newFrame = true;
        }

        if (waitKey(1) == 27) break;  // Exit on ESC key
    }

    cap.release();
}

int main() {
    const string classFile = "/home/mugdha/akshay/object_code/yolo4_tiny/models/coco.names";  
    const string modelConfiguration = "/home/mugdha/akshay/object_code/yolo4_tiny/models/yolov4-tiny.cfg";
    const string modelWeights = "/home/mugdha/akshay/object_code/yolo4_tiny/models/yolov4-tiny.weights";

    const float confThreshold = 0.5f;  // Confidence threshold
    const float nmsThreshold = 0.4f;   // Non-maxima suppression threshold

    dnn::Net net;
    initializeYOLO(net, modelConfiguration, modelWeights);

    vector<string> classNames = loadClassNames(classFile);

    Mat frameCamera1, frameCamera2;
    bool newFrame1 = false, newFrame2 = false;

    // Start threads for both cameras
    thread camera1Thread(processCamera, 0, ref(net), ref(classNames), confThreshold, nmsThreshold, ref(frameCamera1), ref(newFrame1));
    thread camera2Thread(processCamera, 2, ref(net), ref(classNames), confThreshold, nmsThreshold, ref(frameCamera2), ref(newFrame2));

    while (true) {
        // Synchronize frames between threads and display them in the main thread
        if (newFrame1 && newFrame2) {
            lock_guard<mutex> lock(frameMutex);

            // Resize second frame to match the first frame's dimensions
            Mat resizedFrame2;
            resize(frameCamera2, resizedFrame2, Size(frameCamera1.cols, frameCamera1.rows));

            // Combine frames horizontally
            Mat combinedFrame;
            hconcat(frameCamera1, resizedFrame2, combinedFrame);

            // Show combined frame
            imshow("Combined YOLOv4-tiny", combinedFrame);

            newFrame1 = newFrame2 = false;  // Reset the flags
        }

        // Exit on ESC key
        if (waitKey(1) == 27) break;
    }

    // Wait for both threads to finish
    camera1Thread.join();
    camera2Thread.join();

    return 0;
}
