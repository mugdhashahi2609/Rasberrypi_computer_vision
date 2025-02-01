#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int main() {
    // File paths to YOLO configuration, weights, and class names
    const std::string modelConfiguration = "/home/mugdha/Rasberrypi_computer_vision/object_code_new/yolov4-tiny.cfg";
    const std::string modelWeights = "/home/mugdha/Rasberrypi_computer_vision/object_code_new/yolov4-tiny.weights";
    const std::string classFile = "/home/mugdha/Rasberrypi_computer_vision/object_code_new/coco.names";

    // Load YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    if (net.empty()) {
        std::cerr << "Error: Could not load YOLO model\n";
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::vector<std::string> classNames;
    std::ifstream classNamesFile(classFile);
    std::string line;
    while (std::getline(classNamesFile, line)) {
        classNames.push_back(line);
    }

    // Open both cameras
    cv::VideoCapture cap1(0, cv::CAP_V4L2);  // Camera 1 (index 0)
    cv::VideoCapture cap2(2, cv::CAP_V4L2);  // Camera 2 (index 2)

    if (!cap1.isOpened() || !cap2.isOpened()) {
        std::cerr << "Error: Could not open cameras.\n";
        return -1;
    }

    // Set resolution (use standard 416x416 for YOLO)
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 416);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 416);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 416);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 416);

    // Main loop to display the camera feed
    while (true) {
        cv::Mat frame1, frame2;
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty()) {
            std::cerr << "Warning: Empty frame from camera.\n";
            continue;
        }

        // Preprocess frames for YOLO detection (frame1)
        cv::Mat blob1;
        cv::dnn::blobFromImage(frame1, blob1, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob1);

        // Run inference for camera 1 (frame1)
        std::vector<cv::Mat> outs1;
        net.forward(outs1, net.getUnconnectedOutLayersNames());

        // Post-process results for camera 1 (drawing bounding boxes and labels)
        for (size_t i = 0; i < outs1.size(); ++i) {
            float* data = (float*)outs1[i].data;
            for (int j = 0; j < outs1[i].rows; ++j, data += outs1[i].cols) {
                cv::Mat scores = outs1[i].row(j).colRange(5, outs1[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;

                if (confidence > 0.5) {
                    // Get the bounding box coordinates
                    int centerX = (int)(data[0] * frame1.cols);
                    int centerY = (int)(data[1] * frame1.rows);
                    int width = (int)(data[2] * frame1.cols);
                    int height = (int)(data[3] * frame1.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Draw the bounding box and label on frame1
                    cv::rectangle(frame1, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame1, classNames[classId] + ": " + cv::format("%.2f", confidence), cv::Point(left, top - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Preprocess frames for YOLO detection (frame2)
        cv::Mat blob2;
        cv::dnn::blobFromImage(frame2, blob2, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob2);

        // Run inference for camera 2 (frame2)
        std::vector<cv::Mat> outs2;
        net.forward(outs2, net.getUnconnectedOutLayersNames());

        // Post-process results for camera 2 (drawing bounding boxes and labels)
        for (size_t i = 0; i < outs2.size(); ++i) {
            float* data = (float*)outs2[i].data;
            for (int j = 0; j < outs2[i].rows; ++j, data += outs2[i].cols) {
                cv::Mat scores = outs2[i].row(j).colRange(5, outs2[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;

                if (confidence > 0.5) {
                    // Get the bounding box coordinates
                    int centerX = (int)(data[0] * frame2.cols);
                    int centerY = (int)(data[1] * frame2.rows);
                    int width = (int)(data[2] * frame2.cols);
                    int height = (int)(data[3] * frame2.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // Draw the bounding box and label on frame2
                    cv::rectangle(frame2, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame2, classNames[classId] + ": " + cv::format("%.2f", confidence), cv::Point(left, top - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Resize frames to the same size
        cv::resize(frame1, frame1, cv::Size(416, 416));
        cv::resize(frame2, frame2, cv::Size(416, 416));

        // Ensure both frames have the same type (CV_8UC3)
        if (frame1.type() != CV_8UC3) {
            frame1.convertTo(frame1, CV_8UC3);
        }
        if (frame2.type() != CV_8UC3) {
            frame2.convertTo(frame2, CV_8UC3);
        }

        // Combine both frames side by side
        cv::Mat combined;
        cv::hconcat(frame1, frame2, combined);  // Horizontal concatenation

        // Show the combined frame
        cv::imshow("Camera 1 and Camera 2", combined);

        // Exit on ESC key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
