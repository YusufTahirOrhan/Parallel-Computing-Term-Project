#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <cmath>

// ---------------------------------------------------------------------------------
// 1) Calculate gradient directions (0..8) for each pixel
// ---------------------------------------------------------------------------------
void calculateGradientSerial(const cv::Mat& image, cv::Mat& gradientLabels) {
    // gradientLabels should be 8-bit since directions are 0..8
    gradientLabels = cv::Mat::zeros(image.size(), CV_8U);

    // We assume 'image' is a 16-bit grayscale (CV_16U)
    // We'll iterate from 1..cols-1 and 1..rows-1 to avoid boundary issues
    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            // Center pixel intensity
            uint16_t center = image.at<uint16_t>(y, x);

            // 8 neighbors
            uint16_t neighbors[8] = {
                image.at<uint16_t>(y - 1, x),     // N
                image.at<uint16_t>(y - 1, x + 1), // NE
                image.at<uint16_t>(y,     x + 1), // E
                image.at<uint16_t>(y + 1, x + 1), // SE
                image.at<uint16_t>(y + 1, x),     // S
                image.at<uint16_t>(y + 1, x - 1), // SW
                image.at<uint16_t>(y,     x - 1), // W
                image.at<uint16_t>(y - 1, x - 1)  // NW
            };

            // If the pixel is lighter (i.e. smaller intensity value) than all neighbors, direction=0
            // Otherwise, it points to the lightest neighbor
            uint16_t minValue = center;
            int direction = 0; // 0 means local extremum (lighter than neighbors)
            for (int i = 0; i < 8; ++i) {
                if (neighbors[i] < minValue) {
                    minValue = neighbors[i];
                    direction = i + 1; // directions from 1..8
                }
            }
            gradientLabels.at<uint8_t>(y, x) = static_cast<uint8_t>(direction);
        }
    }
}

// ---------------------------------------------------------------------------------
// 2) For each pixel, compute the new direction based on 7x7 neighborhood average
// ---------------------------------------------------------------------------------
void calculate7x7NeighborhoodSerial(const cv::Mat& gradient, cv::Mat& newLabels) {
    // newLabels is also 8-bit, each pixel in [0..8]
    newLabels = cv::Mat::zeros(gradient.size(), CV_8U);

    // We skip a 3-pixel margin because we need a 7x7 area
    for (int y = 3; y < gradient.rows - 3; ++y) {
        for (int x = 3; x < gradient.cols - 3; ++x) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            // Accumulate direction vectors in 7x7 neighborhood
            for (int ky = -3; ky <= 3; ++ky) {
                for (int kx = -3; kx <= 3; ++kx) {
                    int direction = gradient.at<uint8_t>(y + ky, x + kx);

                    // Convert direction into (dx, dy)
                    switch (direction) {
                    case 1: sumY -= 1.0f;               break; // N
                    case 2: sumY -= 1.0f; sumX += 1.0f;  break; // NE
                    case 3: sumX += 1.0f;               break; // E
                    case 4: sumY += 1.0f; sumX += 1.0f;  break; // SE
                    case 5: sumY += 1.0f;               break; // S
                    case 6: sumY += 1.0f; sumX -= 1.0f;  break; // SW
                    case 7: sumX -= 1.0f;               break; // W
                    case 8: sumY -= 1.0f; sumX -= 1.0f;  break; // NW
                    default: break; // direction=0 means no movement
                    }
                }
            }

            // Calculate angle from sumX,sumY
            float magnitude = std::sqrt(sumX * sumX + sumY * sumY);
            int newDirection = 0;

            if (magnitude > 0.0f) {
                float angle = std::atan2(sumY, sumX) * 180.0f / CV_PI;

                // 8 directional sectors
                // E=3, NE=2, N=1, NW=8, W=7, SW=6, S=5, SE=4
                if (angle >= -22.5f && angle < 22.5f)   newDirection = 3; // E
                else if (angle >= 22.5f && angle < 67.5f)   newDirection = 2; // NE
                else if (angle >= 67.5f && angle < 112.5f)   newDirection = 1; // N
                else if (angle >= 112.5f && angle < 157.5f)   newDirection = 8; // NW
                else if (angle >= -67.5f && angle < -22.5f)   newDirection = 4; // SE
                else if (angle >= -112.5f && angle < -67.5f)   newDirection = 5; // S
                else if (angle >= -157.5f && angle < -112.5f)  newDirection = 6; // SW
                else                                          newDirection = 7; // W
            }

            newLabels.at<uint8_t>(y, x) = static_cast<uint8_t>(newDirection);
        }
    }
}

// ---------------------------------------------------------------------------------
// 3) For each pixel (x,y), trace a path until:
//    - direction=0, or
//    - out of bounds, or
//    - step limit reached (to avoid infinite loops)
//    Then record (startX, startY) -> (endX, endY).
//    This matches the parallel approach of "one path per pixel" exactly.
// ---------------------------------------------------------------------------------
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>
findEndpointsSerial(const cv::Mat& directions)
{
    // We'll store (startX, startY) and (endX, endY) for each pixel
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> endpoints;
    endpoints.reserve(directions.rows * directions.cols);

    int width = directions.cols;
    int height = directions.rows;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Each pixel is a new path start
            int startX = x;
            int startY = y;
            int currentX = x;
            int currentY = y;

            // We add a step limit to avoid infinite loops if there's a cycle
            int steps = 0;
            int maxSteps = width * height; // a safe upper bound

            while (true) {
                steps++;
                if (steps > maxSteps) {
                    // break to avoid infinite cycles
                    break;
                }

                // direction=0 => stop
                int direction = directions.at<uint8_t>(currentY, currentX);
                if (direction == 0) {
                    break;
                }

                // compute next pixel
                int nextX = currentX;
                int nextY = currentY;

                switch (direction) {
                case 1:  nextY--;            break; // N
                case 2:  nextY--; nextX++;   break; // NE
                case 3:  nextX++;            break; // E
                case 4:  nextY++; nextX++;   break; // SE
                case 5:  nextY++;            break; // S
                case 6:  nextY++; nextX--;   break; // SW
                case 7:  nextX--;            break; // W
                case 8:  nextY--; nextX--;   break; // NW
                default: break; // 0 or invalid => break
                }

                // if out of bounds, stop
                if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
                    break;
                }

                // move forward
                currentX = nextX;
                currentY = nextY;
            }

            // after the loop ends, (currentX, currentY) is our endpoint
            int endX = currentX;
            int endY = currentY;

            endpoints.push_back({ {startX, startY}, {endX, endY} });
        }
    }
    return endpoints;
}

// ---------------------------------------------------------------------------------
// Main serial processing function
// ---------------------------------------------------------------------------------
void processImageSerial(const cv::Mat& image, const std::string& outputPath) {
    // (1) Calculate initial gradient directions
    cv::Mat gradient;
    calculateGradientSerial(image, gradient);

    // (2) Calculate 7x7 updated directions
    cv::Mat newLabels;
    calculate7x7NeighborhoodSerial(gradient, newLabels);

    // (3) For each pixel, find (start->end)
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> endpoints
        = findEndpointsSerial(newLabels);

    // (4) Write results to a text file
    std::ofstream outputFile(outputPath + ".txt");
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file for writing paths: "
            << outputPath << ".txt\n";
        return;
    }

    // We have exactly one result per pixel => total = rows*cols lines
    for (const auto& e : endpoints) {
        const auto& start = e.first;  // (startX, startY)
        const auto& end = e.second; // (endX, endY)
        outputFile << "(" << start.first << "," << start.second << ") "
            << "(" << end.first << "," << end.second << ")\n";
    }

    outputFile.close();
    std::cout << "Gradient endpoints saved to '"
        << outputPath << ".txt'." << std::endl;
}
