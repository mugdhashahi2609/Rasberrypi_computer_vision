#ifndef YOLO_UTILS_HPP
#define YOLO_UTILS_HPP

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

// Load class names from a file
std::vector<std::string> loadClassNames(const std::string& classFile);

// Draw bounding boxes around detected objects
void drawPredictions(int classId, float confidence, int left, int top, int right, int bottom, 
                     cv::Mat& frame, const std::vector<std::string>& classNames);

#endif // YOLO_UTILS_HPP
