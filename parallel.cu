#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "parallel.h"
#include "device_launch_parameters.h"
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA kernel to calculate gradients
__global__ void calculateGradientCUDA(const uint16_t* d_image, uint8_t* d_gradient, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Boundary check
	if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
		int idx = y * width + x;
		uint16_t center = d_image[idx];

		// Read 8 neighbors
		uint16_t neighbors[8] = {
			d_image[idx - width],        // N
			d_image[idx - width + 1],    // NE
			d_image[idx + 1],            // E
			d_image[idx + width + 1],    // SE
			d_image[idx + width],        // S
			d_image[idx + width - 1],    // SW
			d_image[idx - 1],            // W
			d_image[idx - width - 1]     // NW
		};

		// According to the problem statement:
		// If the pixel is 'lighter' (with a smaller intensity value) than all neighbors, direction=0.
		// Otherwise, point to the lightest neighbor (the one with the smallest intensity).
		uint16_t minValue = center;
		int direction = 0;
		for (int i = 0; i < 8; ++i) {
			if (neighbors[i] < minValue) {
				minValue = neighbors[i];
				direction = i + 1; // (1..8)
			}
		}
		d_gradient[idx] = direction;
	}
}

// CUDA kernel to calculate updated labels in a 7x7 neighborhood
__global__ void calculate7x7NeighborhoodCUDA(const uint8_t* d_gradient, uint8_t* d_newLabels, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Boundary check (3 pixels margin for 7x7)
	if (x >= 3 && x < width - 3 && y >= 3 && y < height - 3) {
		float sumX = 0.0f;
		float sumY = 0.0f;

		// Accumulate direction vectors within the 7x7 neighborhood
		for (int ky = -3; ky <= 3; ++ky) {
			for (int kx = -3; kx <= 3; ++kx) {
				int neighborIdx = (y + ky) * width + (x + kx);
				int direction = d_gradient[neighborIdx];

				// Convert direction (1..8) into a unit vector
				switch (direction) {
				case 1: sumY -= 1.0f;          break; // N
				case 2: sumY -= 1.0f; sumX += 1.0f; break; // NE
				case 3: sumX += 1.0f;          break; // E
				case 4: sumY += 1.0f; sumX += 1.0f; break; // SE
				case 5: sumY += 1.0f;          break; // S
				case 6: sumY += 1.0f; sumX -= 1.0f; break; // SW
				case 7: sumX -= 1.0f;          break; // W
				case 8: sumY -= 1.0f; sumX -= 1.0f; break; // NW
				}
			}
		}

		// Calculate magnitude & angle
		float magnitude = sqrtf(sumX * sumX + sumY * sumY);
		int newDirection = 0;

		if (magnitude > 0.0f) {
			float angle = atan2f(sumY, sumX) * 180.0f / M_PI;
			// Determine direction based on angle
			if (angle >= -22.5f && angle < 22.5f)    newDirection = 3; // E
			else if (angle >= 22.5f && angle < 67.5f)    newDirection = 2; // NE
			else if (angle >= 67.5f && angle < 112.5f)   newDirection = 1; // N
			else if (angle >= 112.5f && angle < 157.5f)   newDirection = 8; // NW
			else if (angle >= -67.5f && angle < -22.5f)   newDirection = 4; // SE
			else if (angle >= -112.5f && angle < -67.5f)   newDirection = 5; // S
			else if (angle >= -157.5f && angle < -112.5f)  newDirection = 6; // SW
			else                                          newDirection = 7; // W
		}

		int idx = y * width + x;
		d_newLabels[idx] = newDirection;
	}
}

// ---------------------------------------------------------------------------------
// CUDA kernel to find endpoints for each pixel. Instead of storing
// the entire path, we only store (startX, startY, endX, endY).
// 
// Added a 'step limit' to avoid infinite loops if there's a cycle.
// ---------------------------------------------------------------------------------
__global__ void findEndpointsCUDA(const uint8_t* d_directions, int* d_endpoints, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;

	// Store the starting pixel (beginning of the path)
	int startX = x;
	int startY = y;

	// We will follow the gradient directions until we reach:
	// - direction=0 (local extremum)
	// - boundary
	// - or step limit (to break a potential cycle)
	int currentX = x;
	int currentY = y;

	// Step limit to avoid infinite loops in case of a cycle
	int steps = 0;
	const int maxSteps = width * height; // or a smaller heuristic, e.g. 2*(width+height)

	while (true) {
		// If steps exceed this limit, break to avoid infinite loops
		if (++steps > maxSteps) {
			break;
		}

		int direction = d_directions[currentY * width + currentX];
		if (direction == 0) {
			// local minimum or no smaller neighbor
			break;
		}

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
		default:
			// Unexpected direction; break out.
			break;
		}

		// If we go out of bounds, stop
		if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
			break;
		}

		// Move forward
		currentX = nextX;
		currentY = nextY;
	}

	// Final endpoint
	int endX = currentX;
	int endY = currentY;

	// Write to global memory: each pixel has 4 integers
	// (startX, startY, endX, endY)
	d_endpoints[4 * idx + 0] = startX;
	d_endpoints[4 * idx + 1] = startY;
	d_endpoints[4 * idx + 2] = endX;
	d_endpoints[4 * idx + 3] = endY;
}

 // ---------------------------------------------------------------------------------
 // processTileParallel:
 // Processes one tile using the above kernels.
 // 
 // 1) Gradient calculation
 // 2) 7x7 neighborhood average direction
 // 3) Endpoint detection with step limit
 // ---------------------------------------------------------------------------------
void processTileParallel(const cv::Mat& tile, cv::Mat& newLabels, std::vector<std::vector<std::pair<int, int>>>& paths)
{
	int width = tile.cols;
	int height = tile.rows;

	// Memory sizes
	size_t imageSize = width * height * sizeof(uint16_t);
	size_t gradientSize = width * height * sizeof(uint8_t);
	size_t endpointsSize = width * height * 4 * sizeof(int);

	// Allocate device memory
	uint16_t* d_image = nullptr;
	uint8_t* d_gradient = nullptr;
	uint8_t* d_newLabels = nullptr;
	int* d_endpoints = nullptr;

	cudaMalloc(&d_image, imageSize);
	cudaMalloc(&d_gradient, gradientSize);
	cudaMalloc(&d_newLabels, gradientSize);
	cudaMalloc(&d_endpoints, endpointsSize);

	// Copy tile data to device
	cudaMemcpy(d_image, tile.data, imageSize, cudaMemcpyHostToDevice);
	cudaMemset(d_endpoints, -1, endpointsSize);

	// You can adjust block size according to your GPU
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	// 1) Gradient kernel
	calculateGradientCUDA << <gridSize, blockSize >> > (d_image, d_gradient, width, height);
	cudaDeviceSynchronize();

	// 2) 7x7 neighborhood kernel
	calculate7x7NeighborhoodCUDA << <gridSize, blockSize >> > (d_gradient, d_newLabels, width, height);
	cudaDeviceSynchronize();

	// 3) Find endpoints
	findEndpointsCUDA << <gridSize, blockSize >> > (d_newLabels, d_endpoints, width, height);
	cudaDeviceSynchronize();

	// Copy final labels (for visualization or debugging)
	newLabels = cv::Mat(tile.size(), CV_8U);
	cudaMemcpy(newLabels.data, d_newLabels, gradientSize, cudaMemcpyDeviceToHost);

	// Copy endpoint data
	std::vector<int> h_endpoints(width * height * 4, -1);
	cudaMemcpy(h_endpoints.data(), d_endpoints, endpointsSize, cudaMemcpyDeviceToHost);

	// Convert endpoint data to a vector of vectors of pairs
	// each pixel => [(startX, startY), (endX, endY)]
	paths.clear();
	paths.reserve(width * height);

	for (int i = 0; i < width * height; ++i) {
		int startX = h_endpoints[4 * i + 0];
		int startY = h_endpoints[4 * i + 1];
		int endX = h_endpoints[4 * i + 2];
		int endY = h_endpoints[4 * i + 3];

		if (startX >= 0 && startY >= 0) {
			std::vector<std::pair<int, int>> path(2);
			path[0] = { startX, startY };
			path[1] = { endX,   endY };
			paths.push_back(path);
		}
	}

	// Free device memory
	cudaFree(d_image);
	cudaFree(d_gradient);
	cudaFree(d_newLabels);
	cudaFree(d_endpoints);
}

 // ---------------------------------------------------------------------------------
 // processImageWithTiling:
 // Splits the large image into tiles, processes each tile on the GPU,
 // and combines / logs results.
 // ---------------------------------------------------------------------------------
void processImageWithTiling(const cv::Mat& image, const std::string& outputPath, int tileSize, int overlap)
{
	int width = image.cols;
	int height = image.rows;

	// Prepare an output image (just to store new directions if needed)
	cv::Mat combinedNewLabels = cv::Mat::zeros(image.size(), CV_8U);

	// For storing all paths from all tiles
	std::vector<std::vector<std::pair<int, int>>> allPaths;
	allPaths.reserve(width * height);

	// Iterate over tiles
	for (int y = 0; y < height; y += tileSize) {
		for (int x = 0; x < width; x += tileSize) {
			// Calculate tile boundaries including overlap
			int startX = std::max(0, x - overlap);
			int startY = std::max(0, y - overlap);
			int endX = std::min(width, x + tileSize + overlap);
			int endY = std::min(height, y + tileSize + overlap);

			// Extract the tile region from the original image
			cv::Rect tileRegion(startX, startY, endX - startX, endY - startY);
			cv::Mat extendedTile = image(tileRegion);

			// Process the tile on GPU
			cv::Mat tileNewLabels;
			std::vector<std::vector<std::pair<int, int>>> tilePaths;
			processTileParallel(extendedTile, tileNewLabels, tilePaths);

			// Copy tile result back to the combined image (excluding overlap)
			cv::Rect processingRegion(
				x - startX,
				y - startY,
				std::min(tileSize, width - x),
				std::min(tileSize, height - y)
			);
			cv::Mat tileResultROI = tileNewLabels(processingRegion);
			tileResultROI.copyTo(combinedNewLabels(cv::Rect(x, y, tileResultROI.cols, tileResultROI.rows)));

			// Adjust paths to global coordinates (because tilePaths are local)
			for (auto& path : tilePaths) {
				for (auto& point : path) {
					point.first += startX; // offset X
					point.second += startY; // offset Y
				}
			}
			allPaths.insert(allPaths.end(), tilePaths.begin(), tilePaths.end());
		}
	}

	// Save (start,end) pairs to a text file
	std::ofstream outputFile(outputPath + ".txt");
	if (outputFile.is_open()) {
		for (const auto& p : allPaths) {
			if (p.size() == 2) {
				outputFile << "(" << p[0].first << "," << p[0].second << ") "
					<< "(" << p[1].first << "," << p[1].second << ")\n";
			}
		}
		outputFile.close();
		std::cout << "Gradient paths saved as '" << outputPath << ".txt'." << std::endl;
	}
	else {
		std::cerr << "Could not open file for writing: " << outputPath << ".txt" << std::endl;
	}
}
