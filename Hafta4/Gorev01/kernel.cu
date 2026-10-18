
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

// CUDA: 3x3 Prewitt filtresi
__global__ void prewittKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < height - 1)
    {
        int gx = 0;

        // [-1 0 1]
        // [-1 0 1]
        // [-1 0 1]
        for (int ky = -1; ky <= 1; ky++)
        {
            gx +=
                -(int)input[(y + ky) * width + (x - 1)]
                + (int)input[(y + ky) * width + (x + 1)];
        }

        int magnitude = abs(gx);

        if (magnitude > 255)
            magnitude = 255;

        output[y * width + x] =
            (unsigned char)magnitude;
    }
}

int main()
{
    std::string imagePath = "duvar.jpeg";


    cv::Mat image = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cout << "Resim okunamadi: "
            << imagePath << std::endl;
        return -1;
    }

    cv::resize(image, image, cv::Size(1024, 768));

    cv::Mat blurred;
    cv::GaussianBlur(
        image,
        blurred,
        cv::Size(3, 3),
        0.8
    );

    int width = blurred.cols;
    int height = blurred.rows;

    size_t imageSize =
        (size_t)width * height * sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

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

    status = cudaMemcpy(
        d_input,
        blurred.data,
        imageSize,
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess)
    {
        std::cout << "Goruntu kopyalama hatasi: "
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

    status = cudaMemset(d_output, 0, imageSize);

    if (status != cudaSuccess)
    {
        std::cout << "Bellek sifirlama hatasi!" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    prewittKernel << <gridSize, blockSize >> > (
        d_input,
        d_output,
        width,
        height
        );

    status = cudaGetLastError();

    if (status == cudaSuccess)
        status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        std::cout << "Prewitt CUDA hatasi: "
            << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();

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
        return -1;
    }

    double elapsed =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    cv::imwrite("cuda_prewitt_sonuc.png", result);

    std::cout << "Gauss + Prewitt islemi tamamlandi!" << std::endl;
    std::cout << "Goruntu: " << width << " x " << height << std::endl;
    std::cout << "Prewitt kernel suresi: "
        << elapsed << " ms" << std::endl;
    std::cout << "Kaydedilen dosya: cuda_prewitt_sonuc.png"
        << std::endl;

    cv::imshow("Orijinal", image);
    cv::imshow("Gauss ile Yumusatilmis", blurred);
    cv::imshow("CUDA Prewitt Gx", result);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}