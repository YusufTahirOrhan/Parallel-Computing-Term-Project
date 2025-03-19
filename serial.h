#ifndef SERIAL_H
#define SERIAL_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

// Prototypes for serial processing functions
void calculateGradientSerial(const cv::Mat& image, cv::Mat& newLabels);
void calculate7x7NeighborhoodSerial(const cv::Mat& gradient, cv::Mat& newLabels);
std::vector<std::vector<std::pair<int, int>>> findGradientPaths(const cv::Mat& gradient);
void processImageSerial(const cv::Mat& image, const std::string& outputPath);

#endif // SERIAL_H
