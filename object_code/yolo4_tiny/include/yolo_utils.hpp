#ifndef YOLO_UTILS_HPP
#define YOLO_UTILS_HPP

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

// Load class names from a file
std::vector<std::string> loadClassNames(const std::string& classFile);

// Initialize the YOLO model with configuration and weights files
void initializeYOLO(cv::dnn::Net& net, const std::string& cfgFile, const std::string& weightsFile);

// Detect objects in the frame
void detectObjects(const cv::Mat& frame, cv::dnn::Net& net,
                   std::vector<cv::Rect>& boxes, std::vector<int>& classIds,
                   std::vector<float>& confidences, float confThreshold = 0.5, float nmsThreshold = 0.4);

// Draw bounding boxes and labels on the frame
void drawDetections(cv::Mat& frame, const std::vector<cv::Rect>& boxes,
                    const std::vector<int>& classIds, const std::vector<float>& confidences,
                    const std::vector<std::string>& classNames);

#endif // YOLO_UTILS_HPP
