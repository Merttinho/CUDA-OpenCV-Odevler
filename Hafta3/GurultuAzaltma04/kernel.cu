
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA hatasi: "                      \
                      << cudaGetErrorString(err) << std::endl;\
            return -1;                                        \
        }                                                     \
    } while (0)


__global__ void meanFilter3x3(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int sum = 0;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));

            sum += input[ny * width + nx];
        }
    }

    output[y * width + x] =
        static_cast<unsigned char>((sum + 4) / 9);
}



__global__ void medianFilter5x5(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned char values[25];
    int index = 0;

    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));

            values[index++] = input[ny * width + nx];
        }
    }

    for (int i = 1; i < 25; i++)
    {
        unsigned char key = values[i];
        int j = i - 1;

        while (j >= 0 && values[j] > key)
        {
            values[j + 1] = values[j];
            j--;
        }

        values[j + 1] = key;
    }

    output[y * width + x] = values[12];
}


int main()
{

    std::string meanImagePath =
        "C:/Users/USER/Desktop/Hafta3/GurultuAzaltma04/siyahbeyaz.jpeg";

    std::string medianImagePath =
        "C:/Users/USER/Desktop/Hafta3/GurultuAzaltma04/salt_pepper.jpg";



    cv::Mat meanImage =
        cv::imread(meanImagePath, cv::IMREAD_GRAYSCALE);

    if (meanImage.empty())
    {
        std::cerr << "3x3 gorevi icin resim okunamadi: "
            << meanImagePath << std::endl;
        return -1;
    }

    cv::resize(meanImage, meanImage, cv::Size(1024, 768));
    meanImage = meanImage.clone();

    int width = meanImage.cols;
    int height = meanImage.rows;
    size_t bytes = width * height * sizeof(unsigned char);

    cv::Mat cpuMean(height, width, CV_8UC1);
    cv::Mat gpuMean(height, width, CV_8UC1);

    auto cpuStart = std::chrono::high_resolution_clock::now();

    cv::blur(meanImage, cpuMean, cv::Size(3, 3));

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMeanMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, meanImage.data, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    meanFilter3x3 << <grid, block >> > (
        d_input, d_output, width, height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMeanMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(
        &gpuMeanMs, start, stop));

    CUDA_CHECK(cudaMemcpy(
        gpuMean.data, d_output, bytes, cudaMemcpyDeviceToHost));

    cv::imwrite("mean_cpu.png", cpuMean);
    cv::imwrite("mean_cuda.png", gpuMean);

    std::cout << "\n===== 3x3 ORTALAMA FILTRESI =====\n";
    std::cout << "CPU (OpenCV blur): "
        << cpuMeanMs << " ms\n";
    std::cout << "CUDA kernel: "
        << gpuMeanMs << " ms\n";

    cv::imshow("Orijinal - Ortalama", meanImage);
    cv::imshow("CPU 3x3 Ortalama", cpuMean);
    cv::imshow("CUDA 3x3 Ortalama", gpuMean);

    cudaFree(d_input);
    cudaFree(d_output);


    cv::Mat medianImage =
        cv::imread(medianImagePath, cv::IMREAD_GRAYSCALE);

    if (medianImage.empty())
    {
        std::cerr << "Salt-and-Pepper resmi okunamadi: "
            << medianImagePath << std::endl;

        cv::waitKey(0);
        return -1;
    }

    cv::resize(medianImage, medianImage, cv::Size(1024, 768));
    medianImage = medianImage.clone();

    width = medianImage.cols;
    height = medianImage.rows;
    bytes = width * height * sizeof(unsigned char);

    cv::Mat cpuMedian(height, width, CV_8UC1);
    cv::Mat gpuMedian(height, width, CV_8UC1);

    cpuStart = std::chrono::high_resolution_clock::now();

    cv::medianBlur(medianImage, cpuMedian, 5);

    cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMedianMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, medianImage.data, bytes, cudaMemcpyHostToDevice));

    grid = dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    CUDA_CHECK(cudaEventRecord(start));

    medianFilter5x5 << <grid, block >> > (
        d_input, d_output, width, height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMedianMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(
        &gpuMedianMs, start, stop));

    CUDA_CHECK(cudaMemcpy(
        gpuMedian.data, d_output, bytes, cudaMemcpyDeviceToHost));

    cv::imwrite("median_cpu.png", cpuMedian);
    cv::imwrite("median_cuda.png", gpuMedian);

    std::cout << "\n===== 5x5 MEDYAN FILTRESI =====\n";
    std::cout << "CPU (OpenCV medianBlur): "
        << cpuMedianMs << " ms\n";
    std::cout << "CUDA kernel: "
        << gpuMedianMs << " ms\n";

    cv::imshow("Salt and Pepper - Orijinal", medianImage);
    cv::imshow("CPU 5x5 Medyan", cpuMedian);
    cv::imshow("CUDA 5x5 Medyan", gpuMedian);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}