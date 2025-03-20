# Software Development for Parallel Computers Term Project

## Problem Statement

Design and implement an image processing algorithm for 16-bit grayscale images that performs the following three steps:

1. **Gradient Calculation:**  
   For each pixel, compare its intensity to that of its 8 immediate neighbors. If the pixel is lighter than all of its neighbors, assign it a gradient value of 0. Otherwise, assign a directional code (1–8 corresponding to N, NE, E, SE, S, SW, W, NW) based on the neighbor with the smallest intensity.

2. **7×7 Neighborhood Update:**  
   For each pixel, refine its gradient direction by considering a 7×7 neighborhood. Convert the directional codes into unit vectors, sum them, and compute an average vector. Map this average back to a directional code (1–8) or 0 if the magnitude is negligible.

3. **Path Tracing:**  
   Starting from each pixel, repeatedly follow the computed gradient directions until a pixel with a gradient value of 0 is reached or the image boundary is encountered. Only the start and end coordinates of the traced path are stored, reducing memory usage.

The primary goal is to compare the performance of a serial (CPU-only) implementation with a parallel (GPU/CUDA) implementation across varying image sizes.

## Overview and Design

### Time Complexity
- **Serial Implementation:**  
  The straightforward CPU approach can lead to O(N⁴) time complexity in the worst case because each pixel may perform up to N² steps in path tracing.
- **Parallel Implementation:**  
  Leveraging GPU parallelism via CUDA, where each pixel is processed by an individual thread, drastically reduces the runtime, especially for larger images.

### Memory Management & Tiling
- **Tiling Strategy:**  
  To manage GPU memory efficiently, the image is split into tiles (using sizes like 256 or 1024), with a 3-pixel overlap to account for the 7×7 neighborhood boundaries.
- **Allocation:**  
  Each tile is processed independently, with dedicated buffers for image data, gradient directions, updated labels, and endpoint coordinates. This approach optimizes both memory usage and performance.

## Performance Evaluation

The project was tested on images ranging from 16×16 to 16384×16384 pixels. Below is an illustrative table of measured times (in milliseconds) for both serial and CUDA implementations with different tile sizes:

| Image Size   | Serial (ms)         | CUDA (tileSize=1024, ms) | CUDA (tileSize=256, ms) |
|--------------|---------------------|--------------------------|-------------------------|
| 16×16        | 2 / 2 / 2           | 86 / 85 / 69             | 85 / 91 / 80            |
| 32×32        | 16 / 15 / 15        | 77 / 119 / 91            | 88 / 93 / 81            |
| 64×64        | 322 / 262 / 330     | 126 / 115 / 105          | 132 / 115 / 130         |
| 128×128      | 4690 / 5249 / 4941   | 254 / 234 / 242          | 258 / 258 / 244         |
| 256×256      | 84769 / 91408 / 92385 | 860 / 876 / 850          | 821 / 825 / 818         |
| 512×512      | 1,611,284           | 4765 / 4805 / 4752       | 3133 / 3218 / 3290      |
| 1024×1024    | Not tested          | 45626 / 35214 / 35579     | 12602 / 12631 / 12758   |
| 16384×16384  | Not tested          | 9,833,751                | 3,957,881               |

*Note: Some larger image tests were omitted for the serial implementation due to impractically long runtimes.*

### Observations
- **Small Images:**  
  Both implementations perform adequately; however, GPU overhead is visible on very small images.
- **Intermediate to Large Images:**  
  The serial approach’s runtime increases dramatically (up to minutes, hours, or even days), while the CUDA implementation scales efficiently.
- **Tile Size Impact:**  
  Experimentation with tile sizes revealed that smaller tiles (256) can sometimes outperform larger ones (1024) due to reduced memory allocation overhead and improved scheduling.

## Conclusion
- The GPU parallel approach demonstrates dramatic speedup over the serial method, especially as image size increases.
- Efficient memory management and tiling are crucial to achieving optimal performance on the GPU.
- The project validates the advantage of parallel computing for large-scale image processing tasks and offers insights for further optimizations (e.g., advanced tiling strategies or skipping redundant computations).

## Testing Environment
- **Operating System:** Windows 11  
- **IDE:** Visual Studio 2022  
- **CUDA Version:** 12.6  
- **Compiler:** nvcc (for CUDA parts) and MSVC (for C++ host code)  
- **Libraries:** OpenCV (for image I/O)
- **CPU:** AMD Ryzen 5 5600x
- **GPU:** NVIDIA GeForce RTX 3070

---
