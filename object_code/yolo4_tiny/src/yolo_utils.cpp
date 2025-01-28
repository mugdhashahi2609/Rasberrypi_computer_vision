#include "yolo_utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to load class names from a file
vector<string> loadClassNames(const string& classFile) {
    vector<string> classNames;
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

// Function to initialize YOLO model
void initializeYOLO(dnn::Net& net, const string& cfgFile, const string& weightsFile) {
    net = readNetFromDarknet(cfgFile, weightsFile);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
}

// Function to perform object detection
void detectObjects(const Mat& frame, dnn::Net& net,
                   vector<Rect>& boxes, vector<int>& classIds,
                   vector<float>& confidences, float confThreshold, float nmsThreshold) {

    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> netOutputs;
    net.forward(netOutputs, net.getUnconnectedOutLayersNames());

    vector<int> outClassIds;
    vector<float> outConfidences;
    vector<Rect> outBoxes;

    for (size_t i = 0; i < netOutputs.size(); ++i) {
        float* data = (float*)netOutputs[i].data;
        for (int j = 0; j < netOutputs[i].rows; ++j, data += netOutputs[i].cols) {
            Mat scores = netOutputs[i].row(j).colRange(5, netOutputs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back((float)confidence);
                classIds.push_back(classIdPoint.x);
            }
        }
    }

    // Apply Non-maxima Suppression to eliminate duplicate boxes
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    vector<Rect> finalBoxes;
    vector<int> finalClassIds;
    vector<float> finalConfidences;

    for (size_t i = 0; i < indices.size(); ++i) {
        finalBoxes.push_back(boxes[indices[i]]);
        finalClassIds.push_back(classIds[indices[i]]);
        finalConfidences.push_back(confidences[indices[i]]);
    }

    boxes = finalBoxes;
    classIds = finalClassIds;
    confidences = finalConfidences;
}

// Function to draw detections on the frame
void drawDetections(Mat& frame, const vector<Rect>& boxes,
                    const vector<int>& classIds, const vector<float>& confidences,
                    const vector<string>& classNames) {

    for (size_t i = 0; i < boxes.size(); ++i) {
        rectangle(frame, boxes[i], Scalar(0, 255, 0), 2);
        string label = format("%.2f", confidences[i]);
        if (!classNames.empty() && classIds[i] < (int)classNames.size()) {
            label = classNames[classIds[i]] + ": " + label;
        }
        putText(frame, label, Point(boxes[i].x, boxes[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }
}