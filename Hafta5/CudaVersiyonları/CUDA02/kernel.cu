#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>
__global__ void myCalcHist(
    unsigned char* image,
    int* hist,
    int totalPixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixels) {
        int pixel = image[idx];
        atomicAdd(&hist[pixel], 1);
    }
}


int main() {
        cv::Mat image = cv::imread("goruntu.jpg");

        if (image.empty()) {
            std::cout << "Goruntu okunamadi!" << std::endl;
            return -1;
        }

        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        
        size_t channelSize = channels[0].rows * channels[0].cols;
        int totalPixels = static_cast<int>(channelSize);

        unsigned char* d_channel;
        int* d_hist;

        cudaMalloc(&d_channel, channelSize);
        cudaMalloc(&d_hist, 256 * sizeof(int));

        int threadsPerBlock = 256;
        int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

        int histB[256], histG[256], histR[256];
        int* histograms[3] = { histB, histG, histR };

        for (int c = 0; c < 3; c++) {
            cudaMemcpy(
                d_channel,
                channels[c].data,
                channelSize,
                cudaMemcpyHostToDevice
            );

            cudaMemset(d_hist, 0, 256 * sizeof(int));

            myCalcHist << <blocks, threadsPerBlock >> > (
                d_channel, d_hist, totalPixels
                );

            cudaDeviceSynchronize();

            cudaMemcpy(
                histograms[c],
                d_hist,
                256 * sizeof(int),
                cudaMemcpyDeviceToHost
            );
        }
        for (int i = 0; i < 256; i++) {
            std::cout << i
                << " | B: " << histB[i]
                << " | G: " << histG[i]
                << " | R: " << histR[i]
                << std::endl;
        }
        cudaFree(d_channel);
        cudaFree(d_hist);
        return 0;
    }