#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>


__global__ void myGrayscale(
    unsigned char* input,
    unsigned char* output,
    int totalPixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int b = input[idx * 3];
        int g = input[idx * 3 + 1];
        int r = input[idx * 3 + 2];

        output[idx] = 0.114f * b
            + 0.587f * g
            + 0.299f * r;
    }
}

int main() {
	cv::Mat image = cv::imread("goruntu.jpg");
	cv::Mat grayImage;
    cv::Mat grayCV;
    cv::cvtColor(image, grayCV, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    int totalPixels = image.rows * image.cols;
    size_t colorBytes = totalPixels * 3 * sizeof(unsigned char);
    size_t grayBytes = totalPixels * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, colorBytes);
    cudaMalloc(&d_output, grayBytes);

    cudaMemcpy(
        d_input,
        image.data,
        colorBytes,
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;
    int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    myGrayscale << <blocks, threadsPerBlock >> > (
        d_input,
        d_output,
        totalPixels
        );

    cudaDeviceSynchronize();

    cv::Mat grayCUDA(image.rows, image.cols, CV_8UC1);

    cudaMemcpy(
        grayCUDA.data,
        d_output,
        grayBytes,
        cudaMemcpyDeviceToHost
    );

    long long toplamHata = 0;
    int farkliPikselSayisi = 0;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int pixelCV = grayCV.at<uchar>(y, x);
            int pixelCUDA = grayCUDA.at<uchar>(y, x);

            int fark = abs(pixelCV - pixelCUDA);

            toplamHata += fark;

            if (fark != 0) {
                farkliPikselSayisi++;
            }
        }
    }

    std::cout << "Toplam mutlak hata: " << toplamHata << std::endl;
    std::cout << "Farkli piksel sayisi: " << farkliPikselSayisi << std::endl;

    cv::imshow("OpenCV Grayscale", grayCV);
    cv::imshow("CUDA Grayscale", grayCUDA);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}