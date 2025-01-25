#include "yolo_utils.hpp"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> loadClassNames(const string& classFile) {
    vector<string> classNames;
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

void drawPredictions(int classId, float confidence, int left, int top, int right, int bottom, 
                     Mat& frame, const vector<string>& classNames) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
    string label = format("%.2f", confidence);
    if (!classNames.empty() && classId < (int)classNames.size()) {
        label = classNames[classId] + ": " + label;
    }
    putText(frame, label, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}
