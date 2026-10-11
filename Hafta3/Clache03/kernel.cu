
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

const int TILES_X = 8;
const int TILES_Y = 8;
const int HIST_SIZE = 256;
const float CLIP_LIMIT = 2.0f;

__global__ void calculateTileHistograms(
    unsigned char* input,
    int* histograms,
    int width,
    int height,
    int tileWidth,
    int tileHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int tileX = x / tileWidth;
        int tileY = y / tileHeight;
        int tileIndex = tileY * TILES_X + tileX;

        int pixel = input[y * width + x];

        atomicAdd(
            &histograms[tileIndex * HIST_SIZE + pixel],
            1
        );
    }
}

__global__ void calculateTileLUT(
    int* histograms,
    unsigned char* lut,
    int tileWidth,
    int tileHeight)
{
    int tileIndex = blockIdx.x;

    if (threadIdx.x == 0)
    {
        int* hist = &histograms[tileIndex * HIST_SIZE];
        unsigned char* map = &lut[tileIndex * HIST_SIZE];

        int totalPixels = tileWidth * tileHeight;
        int clipThreshold = (int)(
            CLIP_LIMIT * totalPixels / HIST_SIZE
            );

        if (clipThreshold < 1)
            clipThreshold = 1;

        int excess = 0;

        for (int i = 0; i < HIST_SIZE; i++)
        {
            if (hist[i] > clipThreshold)
            {
                excess += hist[i] - clipThreshold;
                hist[i] = clipThreshold;
            }
        }

        int batch = excess / HIST_SIZE;
        int remainder = excess % HIST_SIZE;

        for (int i = 0; i < HIST_SIZE; i++)
            hist[i] += batch;

        for (int i = 0; i < remainder; i++)
            hist[i]++;

        int cdf = 0;

        for (int i = 0; i < HIST_SIZE; i++)
        {
            cdf += hist[i];

            int value = (int)(
                (cdf * 255.0f / totalPixels) + 0.5f
                );

            if (value > 255) value = 255;
            if (value < 0) value = 0;

            map[i] = (unsigned char)value;
        }
    }
}

__global__ void applyCLAHE(
    unsigned char* input,
    unsigned char* output,
    unsigned char* lut,
    int width,
    int height,
    int tileWidth,
    int tileHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel = input[y * width + x];

        float tx = x / (float)tileWidth - 0.5f;
        float ty = y / (float)tileHeight - 0.5f;

        int x0 = (int)floorf(tx);
        int y0 = (int)floorf(ty);

        float dx = tx - x0;
        float dy = ty - y0;

        if (x0 < 0) { x0 = 0; dx = 0.0f; }
        if (y0 < 0) { y0 = 0; dy = 0.0f; }

        if (x0 >= TILES_X - 1)
        {
            x0 = TILES_X - 1;
            dx = 0.0f;
        }

        if (y0 >= TILES_Y - 1)
        {
            y0 = TILES_Y - 1;
            dy = 0.0f;
        }

        int x1 = min(x0 + 1, TILES_X - 1);
        int y1 = min(y0 + 1, TILES_Y - 1);

        int i00 = (y0 * TILES_X + x0) * HIST_SIZE + pixel;
        int i10 = (y0 * TILES_X + x1) * HIST_SIZE + pixel;
        int i01 = (y1 * TILES_X + x0) * HIST_SIZE + pixel;
        int i11 = (y1 * TILES_X + x1) * HIST_SIZE + pixel;

        float top =
            lut[i00] * (1.0f - dx) + lut[i10] * dx;

        float bottom =
            lut[i01] * (1.0f - dx) + lut[i11] * dx;

        float value =
            top * (1.0f - dy) + bottom * dy;

        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;

        output[y * width + x] =
            (unsigned char)(value + 0.5f);
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
        std::cout << "Resim okunamadi: "
            << imagePath << std::endl;
        return -1;
    }

    cv::resize(image, image, cv::Size(1024, 768));

    int width = image.cols;
    int height = image.rows;

    int tileWidth = width / TILES_X;
    int tileHeight = height / TILES_Y;

    size_t imageSize =
        (size_t)width * height * sizeof(unsigned char);

    size_t histogramSize =
        (size_t)TILES_X * TILES_Y * HIST_SIZE * sizeof(int);

    size_t lutSize =
        (size_t)TILES_X * TILES_Y * HIST_SIZE *
        sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int* d_histograms = nullptr;
    unsigned char* d_lut = nullptr;

    cudaError_t status;

    status = cudaMalloc((void**)&d_input, imageSize);
    if (status != cudaSuccess) goto cleanup;

    status = cudaMalloc((void**)&d_output, imageSize);
    if (status != cudaSuccess) goto cleanup;

    status = cudaMalloc(
        (void**)&d_histograms, histogramSize
    );
    if (status != cudaSuccess) goto cleanup;

    status = cudaMalloc((void**)&d_lut, lutSize);
    if (status != cudaSuccess) goto cleanup;

    auto totalStart = std::chrono::high_resolution_clock::now();

    status = cudaMemcpy(
        d_input, image.data, imageSize,
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) goto cleanup;

    status = cudaMemset(d_histograms, 0, histogramSize);
    if (status != cudaSuccess) goto cleanup;

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    auto kernelStart = std::chrono::high_resolution_clock::now();

    calculateTileHistograms << <gridSize, blockSize >> > (
        d_input, d_histograms,
        width, height, tileWidth, tileHeight
        );

    status = cudaGetLastError();
    if (status != cudaSuccess) goto cleanup;

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) goto cleanup;

    calculateTileLUT << <TILES_X * TILES_Y, 1 >> > (
        d_histograms, d_lut, tileWidth, tileHeight
        );

    status = cudaGetLastError();
    if (status != cudaSuccess) goto cleanup;

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) goto cleanup;

    applyCLAHE << <gridSize, blockSize >> > (
        d_input, d_output, d_lut,
        width, height, tileWidth, tileHeight
        );

    status = cudaGetLastError();
    if (status != cudaSuccess) goto cleanup;

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) goto cleanup;

    {
        auto kernelEnd = std::chrono::high_resolution_clock::now();

        cv::Mat result(height, width, CV_8UC1);

        status = cudaMemcpy(
            result.data, d_output, imageSize,
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) goto cleanup;

        auto totalEnd = std::chrono::high_resolution_clock::now();

        double kernelTime =
            std::chrono::duration<double, std::milli>(
                kernelEnd - kernelStart
            ).count();

        double totalTime =
            std::chrono::duration<double, std::milli>(
                totalEnd - totalStart
            ).count();

        if (cv::imwrite("cuda_clahe_sonuc.png", result))
            std::cout << "Sonuc kaydedildi: cuda_clahe_sonuc.png"
            << std::endl;
        else
            std::cout << "Sonuc kaydedilemedi!" << std::endl;

        std::cout << "Kernel suresi: "
            << kernelTime << " ms" << std::endl;

        std::cout << "Toplam sure: "
            << totalTime << " ms" << std::endl;

        cv::imshow("Orijinal", image);
        cv::imshow("CUDA CLAHE", result);
        cv::waitKey(0);
    }

cleanup:
    if (status != cudaSuccess)
    {
        std::cout << "CUDA hatasi: "
            << cudaGetErrorString(status) << std::endl;
    }

    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_histograms) cudaFree(d_histograms);
    if (d_lut) cudaFree(d_lut);

    return status == cudaSuccess ? 0 : -1;
}