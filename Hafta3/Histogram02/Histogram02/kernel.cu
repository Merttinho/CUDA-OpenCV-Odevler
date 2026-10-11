
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

__global__ void calculateHistogram(
    unsigned char* input,
    int* histogram,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int pixel = (int)input[index];

        atomicAdd(&histogram[pixel], 1);
    }
}


__global__ void calculateCDF(
    int* histogram,
    int* cdf)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int sum = 0;

        for (int i = 0; i < 256; i++)
        {
            sum += histogram[i];
            cdf[i] = sum;
        }
    }
}

__global__ void equalizeImage(
    unsigned char* input,
    unsigned char* output,
    int* cdf,
    int width,
    int height,
    int totalPixels,
    int cdfMin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int pixel = (int)input[index];

        int newPixel = pixel;

        if (totalPixels > cdfMin && cdfMin >= 0)
        {
            float value =
                ((float)(cdf[pixel] - cdfMin) /
                    (float)(totalPixels - cdfMin)) * 255.0f;

            if (value < 0.0f) value = 0.0f;
            if (value > 255.0f) value = 255.0f;

            newPixel = (int)(value + 0.5f);
        }

        output[index] = (unsigned char)newPixel;
    }
}

int main()
{
    std::string imagePath = "siyahbeyaz.jpeg";

    cv::Mat image = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cout << "Resim okunamadi: " << imagePath << std::endl;
        return -1;
    }

    cv::resize(image, image, cv::Size(1024, 768));

    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;

    std::cout << "Resim: " << width << " x " << height << std::endl;

    size_t imageSize = (size_t)totalPixels * sizeof(unsigned char);
    size_t histogramSize = 256 * sizeof(int);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int* d_histogram = nullptr;
    int* d_cdf = nullptr;

    cudaError_t status;

    status = cudaMalloc((void**)&d_input, imageSize);
    if (status != cudaSuccess)
    {
        std::cout << "Input bellek hatasi: "
            << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    status = cudaMalloc((void**)&d_output, imageSize);
    if (status != cudaSuccess)
    {
        std::cout << "Output bellek hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        return -1;
    }

    status = cudaMalloc((void**)&d_histogram, histogramSize);
    if (status != cudaSuccess)
    {
        std::cout << "Histogram bellek hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    status = cudaMalloc((void**)&d_cdf, histogramSize);
    if (status != cudaSuccess)
    {
        std::cout << "CDF bellek hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        return -1;
    }

  
    status = cudaMemset(d_histogram, 0, histogramSize);
    if (status != cudaSuccess)
    {
        std::cout << "Histogram sifirlama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    auto totalStart = std::chrono::high_resolution_clock::now();


    status = cudaMemcpy(
        d_input,
        image.data,
        imageSize,
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess)
    {
        std::cout << "Resim kopyalama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    auto kernelStart = std::chrono::high_resolution_clock::now();


    calculateHistogram << <gridSize, blockSize >> > (
        d_input,
        d_histogram,
        width,
        height
        );

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "Histogram kernel hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "Histogram calisma hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    calculateCDF << <1, 1 >> > (d_histogram, d_cdf);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "CDF kernel hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "CDF calisma hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }


    int hostHistogram[256];
    int hostCDF[256];

    status = cudaMemcpy(
        hostHistogram,
        d_histogram,
        histogramSize,
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess)
    {
        std::cout << "Histogram geri kopyalama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    status = cudaMemcpy(
        hostCDF,
        d_cdf,
        histogramSize,
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess)
    {
        std::cout << "CDF geri kopyalama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    int cdfMin = 0;

    for (int i = 0; i < 256; i++)
    {
        if (hostHistogram[i] > 0)
        {
            cdfMin = hostCDF[i];
            break;
        }
    }


    equalizeImage << <gridSize, blockSize >> > (
        d_input,
        d_output,
        d_cdf,
        width,
        height,
        totalPixels,
        cdfMin
        );

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "Esitleme kernel hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "Esitleme calisma hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    auto kernelEnd = std::chrono::high_resolution_clock::now();


    cv::Mat result(height, width, CV_8UC1);

    status = cudaMemcpy(
        result.data,
        d_output,
        imageSize,
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess)
    {
        std::cout << "Sonuc kopyalama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return -1;
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double kernelTime =
        std::chrono::duration<double, std::milli>(
            kernelEnd - kernelStart
        ).count();

    double totalTime =
        std::chrono::duration<double, std::milli>(
            totalEnd - totalStart
        ).count();

    if (cv::imwrite("cuda_histogram_sonuc.png", result))
        std::cout << "Sonuc kaydedildi: cuda_histogram_sonuc.png" << std::endl;
    else
        std::cout << "Sonuc kaydedilemedi!" << std::endl;

    std::cout << "Kernel suresi: " << kernelTime << " ms" << std::endl;
    std::cout << "Toplam GPU islem suresi: " << totalTime << " ms" << std::endl;

    cv::imshow("Orijinal", image);
    cv::imshow("CUDA Histogram Esitleme", result);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(d_cdf);

    return 0;
}