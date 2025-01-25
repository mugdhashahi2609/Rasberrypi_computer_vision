#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to load class names from coco.names
vector<string> loadClassNames(const string& classFile) {
    vector<string> classNames;
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) classNames.push_back(line);
    return classNames;
}

// Draw bounding boxes around detected objects
void drawPredictions(int classId, float confidence, int left, int top, int right, int bottom, Mat& frame, const vector<string>& classNames) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
    string label = format("%.2f", confidence);
    if (!classNames.empty() && classId < (int)classNames.size()) {
        label = classNames[classId] + ": " + label;
    }
    putText(frame, label, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}

int main() {
    // Load YOLOv4-tiny model
    const string modelConfiguration = "yolov4-tiny.cfg";
    const string modelWeights = "yolov4-tiny.weights";
    const string classFile = "coco.names";

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load class names
    vector<string> classNames = loadClassNames(classFile);

    // Open camera
    VideoCapture cap(0);  // 0 for default camera
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    Mat frame, blob;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Create input blob for YOLO
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // Perform forward pass
        vector<Mat> netOutputs;
        net.forward(netOutputs, net.getUnconnectedOutLayersNames());

        // Process each detection
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

        // Display frame
        imshow("YOLOv4-tiny Object Detection", frame);

        // Exit on ESC key
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
