#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>
#include <chrono>

__global__ void processImage(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        float pixel = input[index];

    
        float newPixel = pixel * 0.75f + 20.0f;

     
        if (newPixel > 255.0f)
            newPixel = 255.0f;

        if (newPixel < 0.0f)
            newPixel = 0.0f;

        output[index] = (unsigned char)newPixel;
    }
}

int main()
{


    std::string imagePath = "tekkanali.jpg";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cout << "Resim okunamadi!" << std::endl;
        return -1;
    }

    // 1024x768 yap
    cv::resize(image, image, cv::Size(1024, 768));

    int width = image.cols;
    int height = image.rows;

    std::cout << "Resim: "
        << width << " x "
        << height << std::endl;


  

    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaError_t status;

    status = cudaMalloc((void**)&d_input, imageSize);

    if (status != cudaSuccess)
    {
        std::cout << "cudaMalloc input hatasi: "
            << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    status = cudaMalloc((void**)&d_output, imageSize);

    if (status != cudaSuccess)
    {
        std::cout << "cudaMalloc output hatasi: "
            << cudaGetErrorString(status) << std::endl;

        cudaFree(d_input);
        return -1;
    }


 

    auto gpuStart = std::chrono::high_resolution_clock::now();

    status = cudaMemcpy(
        d_input,
        image.data,
        imageSize,
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess)
    {
        std::cout << "CPU -> GPU kopyalama hatasi: "
            << cudaGetErrorString(status) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }




    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    auto kernelStart = std::chrono::high_resolution_clock::now();

    processImage << <gridSize, blockSize >> > (
        d_input,
        d_output,
        width,
        height
        );

    status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        std::cout << "Kernel launch hatasi: "
            << cudaGetErrorString(status) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }

    cudaDeviceSynchronize();

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
        std::cout << "GPU -> CPU kopyalama hatasi: "
            << cudaGetErrorString(status) << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }



    auto gpuEnd = std::chrono::high_resolution_clock::now();

    double kernelTime =
        std::chrono::duration<double, std::milli>(
            kernelEnd - kernelStart
        ).count();

    double totalGpuTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart
        ).count();




    cv::imwrite("cuda_sonuc.jpg", result);

    std::cout << std::endl;
    std::cout << "CUDA islemi tamamlandi!" << std::endl;

    std::cout << "Kernel suresi: "
        << kernelTime
        << " ms" << std::endl;

    std::cout << "Toplam GPU suresi: "
        << totalGpuTime
        << " ms" << std::endl;




    cv::imshow("Orijinal", image);
    cv::imshow("CUDA Sonuc", result);

    cv::waitKey(0);


    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
