#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

__device__ unsigned char blendChannel(
    int start, int end, float t)
{
    return (unsigned char)(start + (end - start) * t + 0.5f);
}

__global__ void colorizeKernel(
    const unsigned char* gray,
    unsigned char* output,
    int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int value = gray[idx];

        int b, g, r;
        float t;

        if (value <= 85) {
            t = value / 85.0f;

            b = blendChannel(128, 255, t);
            g = blendChannel(0, 255, t);
            r = 0;
        }
        else if (value <= 170) {
            t = (value - 85) / 85.0f;

            b = blendChannel(255, 0, t);
            g = 255;
            r = blendChannel(0, 255, t);
        }
        else {
            t = (value - 170) / 85.0f;

            b = 0;
            g = blendChannel(255, 0, t);
            r = 255;
        }

        output[idx * 3] = (unsigned char)b;
        output[idx * 3 + 1] = (unsigned char)g;
        output[idx * 3 + 2] = (unsigned char)r;
    }
}

bool colorizeImage(
    const cv::Mat& gray,
    cv::Mat& colorized)
{
    int totalPixels = gray.rows * gray.cols;

    size_t grayBytes =
        (size_t)totalPixels * sizeof(unsigned char);

    size_t colorBytes =
        (size_t)totalPixels * 3 * sizeof(unsigned char);

    unsigned char* d_gray = nullptr;
    unsigned char* d_output = nullptr;

    if (cudaMalloc((void**)&d_gray, grayBytes) != cudaSuccess)
        return false;

    if (cudaMalloc((void**)&d_output, colorBytes) != cudaSuccess) {
        cudaFree(d_gray);
        return false;
    }

    cudaError_t err = cudaMemcpy(
        d_gray,
        gray.data,
        grayBytes,
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        cudaFree(d_gray);
        cudaFree(d_output);
        return false;
    }

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    colorizeKernel << <blocks, threads >> > (
        d_gray, d_output, totalPixels
        );

    err = cudaGetLastError();

    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cout << "CUDA hatasi: "
            << cudaGetErrorString(err) << std::endl;

        cudaFree(d_gray);
        cudaFree(d_output);
        return false;
    }

    colorized.create(gray.rows, gray.cols, CV_8UC3);

    err = cudaMemcpy(
        colorized.data,
        d_output,
        colorBytes,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_gray);
    cudaFree(d_output);

    return err == cudaSuccess;
}

int main()
{
    std::string imageNames[5] = {
        "image01.jpg",
        "image02.jpg",
        "image03.jpg",
        "image04.jpg",
        "image05.jpg"
    };

    for (int i = 0; i < 5; i++) {
        cv::Mat image = cv::imread(imageNames[i]);

        if (image.empty()) {
            std::cout << imageNames[i]
                << " okunamadi!" << std::endl;
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        if (!gray.isContinuous())
            gray = gray.clone();

        cv::Mat colorized;

        if (!colorizeImage(gray, colorized)) {
            std::cout << imageNames[i]
                << " renklendirilemedi!" << std::endl;
            continue;
        }

        std::string outputName =
            "colorized_0" + std::to_string(i + 1) + ".jpg";

        cv::imwrite(outputName, colorized);

        std::string suffix = std::to_string(i + 1);

        cv::imshow("Original " + suffix, image);
        cv::imshow("CUDA Colorized " + suffix, colorized);

        std::cout << imageNames[i]
            << " islendi. Kaydedilen dosya: "
            << outputName << std::endl;
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
