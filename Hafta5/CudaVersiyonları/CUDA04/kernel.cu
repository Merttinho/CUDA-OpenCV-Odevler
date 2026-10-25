
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

__global__ void grayKernel(
    const unsigned char* input,
    unsigned char* output,
    int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        float b = input[idx * 3];
        float g = input[idx * 3 + 1];
        float r = input[idx * 3 + 2];

        output[idx] = (unsigned char)roundf(
            0.114f * b + 0.587f * g + 0.299f * r
        );
    }
}

__global__ void hsvKernel(
    const unsigned char* input,
    unsigned char* output,
    int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        float b = input[idx * 3] / 255.0f;
        float g = input[idx * 3 + 1] / 255.0f;
        float r = input[idx * 3 + 2] / 255.0f;

        float maxVal = fmaxf(r, fmaxf(g, b));
        float minVal = fminf(r, fminf(g, b));
        float delta = maxVal - minVal;

        float h = 0.0f;
        float s = 0.0f;
        float v = maxVal;

        if (delta != 0.0f) {
            if (maxVal == r) {
                h = 60.0f * ((g - b) / delta);
                if (h < 0.0f)
                    h += 360.0f;
            }
            else if (maxVal == g) {
                h = 60.0f * ((b - r) / delta + 2.0f);
            }
            else {
                h = 60.0f * ((r - g) / delta + 4.0f);
            }

            s = delta / maxVal;
        }

        int H = (int)roundf(h / 2.0f);
        int S = (int)roundf(s * 255.0f);
        int V = (int)roundf(v * 255.0f);

        if (H >= 180) H = 0;

        output[idx * 3] = (unsigned char)H;
        output[idx * 3 + 1] = (unsigned char)S;
        output[idx * 3 + 2] = (unsigned char)V;
    }
}

using ConvertFunc = void (*)(const cv::Mat&, cv::Mat&);

void convertGray(const cv::Mat& image, cv::Mat& output)
{
    int totalPixels = image.rows * image.cols;
    size_t inputBytes = (size_t)totalPixels * 3;
    size_t outputBytes = (size_t)totalPixels;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc((void**)&d_input, inputBytes);
    cudaMalloc((void**)&d_output, outputBytes);

    cudaMemcpy(d_input, image.data, inputBytes,
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    grayKernel << <blocks, threads >> > (
        d_input, d_output, totalPixels);

    cudaDeviceSynchronize();

    output.create(image.rows, image.cols, CV_8UC1);

    cudaMemcpy(output.data, d_output, outputBytes,
        cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void convertHSV(const cv::Mat& image, cv::Mat& output)
{
    int totalPixels = image.rows * image.cols;
    size_t bytes = (size_t)totalPixels * 3;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);

    cudaMemcpy(d_input, image.data, bytes,
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    hsvKernel << <blocks, threads >> > (
        d_input, d_output, totalPixels);

    cudaDeviceSynchronize();

    output.create(image.rows, image.cols, CV_8UC3);

    cudaMemcpy(output.data, d_output, bytes,
        cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

long long compareImages(
    const cv::Mat& reference,
    const cv::Mat& result)
{
    long long totalError = 0;

    for (int y = 0; y < reference.rows; y++) {
        const unsigned char* a = reference.ptr<unsigned char>(y);
        const unsigned char* b = result.ptr<unsigned char>(y);

        int valuesPerRow = reference.cols * reference.channels();

        for (int i = 0; i < valuesPerRow; i++) {
            totalError += std::abs((int)a[i] - (int)b[i]);
        }
    }

    return totalError;
}

int main()
{
    cv::Mat image = cv::imread("goruntu.jpg");

    if (image.empty()) {
        std::cout << "Goruntu okunamadi!" << std::endl;
        return -1;
    }

    if (!image.isContinuous()) {
        image = image.clone();
    }

    cv::Mat grayCV, hsvCV;
    cv::Mat grayCUDA, hsvCUDA;

    cv::cvtColor(image, grayCV, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, hsvCV, cv::COLOR_BGR2HSV);

    ConvertFunc convert = convertGray;
    convert(image, grayCUDA);


    convert = convertHSV;
    convert(image, hsvCUDA);

    std::cout << "Gri tonlama toplam hatasi: "
        << compareImages(grayCV, grayCUDA) << std::endl;

    std::cout << "HSV toplam hatasi: "
        << compareImages(hsvCV, hsvCUDA) << std::endl;

    cv::imshow("OpenCV Gray", grayCV);
    cv::imshow("CUDA Gray", grayCUDA);
    cv::imshow("OpenCV HSV", hsvCV);
    cv::imshow("CUDA HSV", hsvCUDA);

    cv::waitKey(0);

    return 0;
}
