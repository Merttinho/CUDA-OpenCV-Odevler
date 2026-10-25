#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>

__global__ void calcHistogram(unsigned char* image, int* hist, int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int b = image[idx * 3];
        int g = image[idx * 3 + 1];
        int r = image[idx * 3 + 2];

        atomicAdd(&hist[b], 1);
        atomicAdd(&hist[256 + g], 1);
        atomicAdd(&hist[512 + r], 1);
    }
}

int main() {
    cv::Mat image = cv::imread("goruntu.jpg");

    if (image.empty()) {
        std::cout << "Goruntu okunamadi!" << std::endl;
        return -1;
    }

    std::cout << "Goruntu boyutu: "
        << image.cols << "x" << image.rows << std::endl;

    size_t imageSize = image.rows * image.cols * 3;

    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);
    int hist[3][256] = {};
    int* d_hist;
    cudaMalloc(&d_hist, 3 * 256 * sizeof(int));
    cudaMemset(d_hist, 0, 3 * 256 * sizeof(int));

    int totalPixels = image.rows * image.cols;

    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    calcHistogram <<<blocks, threadsPerBlock >>> (d_image, d_hist, totalPixels);

    cudaDeviceSynchronize();

    cudaMemcpy(hist, d_hist, 3 * 256 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; i++) {
        std::cout << i
            << " | B: " << hist[0][i]
            << " | G: " << hist[1][i]
            << " | R: " << hist[2][i]
            << std::endl;
    }
}