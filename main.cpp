#include <opencv2/opencv.hpp>
#include <iostream>
#include "serial.h"
#include "parallel.h"

// Function to load a grayscale image from file
cv::Mat loadImage(const std::string& filePath) {
    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Image could not be loaded!" << std::endl;
        exit(-1);
    }
    if (image.type() != CV_16U) {
        std::cerr << "The image is not 16-bit!" << std::endl;
        exit(-1);
    }
    return image;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: CMP_641_Project.exe [serial|parallel] [image_path]" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string imagePath = argv[2];

    // Load the input image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Could not load the image." << std::endl;
        return -1;
    }

    if (image.type() != CV_16U) {
        std::cerr << "Error: The image must be 16-bit grayscale!" << std::endl;
        return -1;
    }

    if (mode == "serial") {
        std::cout << "Running the serial algorithm..." << std::endl;

        // Measure the execution time
        auto start = std::chrono::high_resolution_clock::now();
        processImageSerial(image, "serial_output");
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate and display execution time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Serial algorithm completed in " << duration.count() << " milliseconds." << std::endl;
    }
    else if (mode == "parallel") {
        std::cout << "Running the parallel algorithm..." << std::endl;

        // Measure the execution time
        auto start = std::chrono::high_resolution_clock::now();
        processImageWithTiling(image, "parallel_output");
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate and display execution time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Parallel algorithm completed in " << duration.count() << " milliseconds." << std::endl;
    }
    else {
        std::cerr << "Invalid mode. Use 'serial' or 'parallel'." << std::endl;
        return -1;
    }

    return 0;
}