
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>


__global__ void findMinMax(
    unsigned char* input,
    int* minMax,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int pixel = (int)input[index];

        atomicMin(&minMax[0], pixel);
        atomicMax(&minMax[1], pixel);
    }
}


__global__ void improveContrast(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int minVal,
    int maxVal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        float pixel = (float)input[index];
        float newPixel;

        if (maxVal == minVal)
        {
            newPixel = pixel;
        }
        else
        {
            newPixel =
                ((pixel - minVal) / (float)(maxVal - minVal)) * 255.0f;
        }

        if (newPixel > 255.0f) newPixel = 255.0f;
        if (newPixel < 0.0f) newPixel = 0.0f;

        output[index] = (unsigned char)(newPixel + 0.5f);
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

    std::cout << "Resim: " << width << " x " << height << std::endl;

    size_t imageSize = (size_t)width * height * sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int* d_minMax = nullptr;

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

    status = cudaMalloc((void**)&d_minMax, 2 * sizeof(int));
    if (status != cudaSuccess)
    {
        std::cout << "MinMax bellek hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    auto totalStart = std::chrono::high_resolution_clock::now();

    status = cudaMemcpy(
        d_input, image.data, imageSize, cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess)
    {
        std::cout << "Resim kopyalama hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

    int initialMinMax[2] = { 255, 0 };

    status = cudaMemcpy(
        d_minMax,
        initialMinMax,
        2 * sizeof(int),
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess)
    {
        std::cout << "Baslangic degerleri kopyalanamadi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

 
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );


    findMinMax << <gridSize, blockSize >> > (
        d_input, d_minMax, width, height
        );

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "MinMax kernel hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "MinMax calisma hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

 
    int minMax[2];

    status = cudaMemcpy(
        minMax,
        d_minMax,
        2 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess)
    {
        std::cout << "MinMax geri kopyalama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

    int minVal = minMax[0];
    int maxVal = minMax[1];

    std::cout << "Minimum piksel: " << minVal << std::endl;
    std::cout << "Maksimum piksel: " << maxVal << std::endl;

  
    auto kernelStart = std::chrono::high_resolution_clock::now();

    improveContrast << <gridSize, blockSize >> > (
        d_input,
        d_output,
        width,
        height,
        minVal,
        maxVal
        );

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "Kontrast kernel hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
        return -1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "Kontrast islemi hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_minMax);
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
        cudaFree(d_minMax);
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


    if (!cv::imwrite("cuda_kontrast_sonuc.png", result))
    {
        std::cout << "Sonuc resmi kaydedilemedi!" << std::endl;
    }
    else
    {
        std::cout << "Sonuc kaydedildi: cuda_kontrast_sonuc.png" << std::endl;
    }

    std::cout << "Kontrast kernel suresi: "
        << kernelTime << " ms" << std::endl;

    std::cout << "Toplam GPU islem suresi: "
        << totalTime << " ms" << std::endl;

    cv::imshow("Orijinal Goruntu", image);
    cv::imshow("CUDA Kontrast Iyilestirme", result);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_minMax);

    return 0;
}