#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "yolo_utils.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    const string modelConfiguration = "../models/yolov4-tiny.cfg";
    const string modelWeights = "../models/yolov4-tiny.weights";
    const string classFile = "../models/coco.names";

    // Load YOLO model
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load class names
    vector<string> classNames = loadClassNames(classFile);

    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    Mat frame, blob;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Prepare the input for YOLO
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // Forward pass
        vector<Mat> netOutputs;
        net.forward(netOutputs, net.getUnconnectedOutLayersNames());

        // Process detections
        for (size_t i = 0; i < netOutputs.size(); ++i) {
            float* data = (float*)netOutputs[i].data;
            for (int j = 0; j < netOutputs[i].rows; ++j, data += netOutputs[i].cols) {
                Mat scores = netOutputs[i].row(j).colRange(5, netOutputs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Draw predictions
                    drawPredictions(classIdPoint.x, (float)confidence, left, top, left + width, top + height, frame, classNames);
                }
            }
        }

        // Show the output
        imshow("YOLOv4-tiny Object Detection", frame);

        // Exit on pressing ESC key
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
