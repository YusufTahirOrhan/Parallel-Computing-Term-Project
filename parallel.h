#ifndef PARALLEL_H
#define PARALLEL_H

#include <opencv2/opencv.hpp>

// Function prototype for the parallel gradient calculation
void processImageWithTiling(const cv::Mat& image, const std::string& outputPath, int tileSize = 256, int overlap = 3);

#endif // PARALLEL_H
