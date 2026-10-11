

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
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

    std::string imagePath =
        "C:/Users/USER/Desktop/Hafta3/MedyanFiltresi05/salt_pepper.jpg";

    cv::Mat image = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cerr << "Resim okunamadi: "
            << imagePath << std::endl;
        return -1;
    }

    cv::resize(image, image, cv::Size(1024, 768));

    int width = image.cols;
    int height = image.rows;
    size_t bytes =
        static_cast<size_t>(width) * height * sizeof(unsigned char);

    cv::Mat cpuResult(height, width, CV_8UC1);
    cv::Mat gpuResult(height, width, CV_8UC1);


    auto cpuStart = std::chrono::high_resolution_clock::now();

    cv::medianBlur(image, cpuResult, 5);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();



    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        image.data,
        bytes,
        cudaMemcpyHostToDevice
    ));


    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    medianFilter5x5 << <grid, block >> > (
        d_input,
        d_output,
        width,
        height
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMs = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &gpuMs,
        start,
        stop
    ));


    CUDA_CHECK(cudaMemcpy(
        gpuResult.data,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));


    cv::imwrite("median_original.png", image);
    cv::imwrite("median_opencv.png", cpuResult);
    cv::imwrite("median_cuda.png", gpuResult);


    cv::Mat difference;
    cv::absdiff(cpuResult, gpuResult, difference);

    int differentPixels = cv::countNonZero(difference);

    std::cout << "\n===== 5x5 MEDYAN FILTRESI =====\n";

    std::cout << "CPU (OpenCV): "
        << cpuMs << " ms\n";

    std::cout << "CUDA kernel: "
        << gpuMs << " ms\n";

    std::cout << "Farkli pixel sayisi: "
        << differentPixels << "\n";

    if (differentPixels == 0)
    {
        std::cout << "Sonuclar tamamen ayni.\n";
    }
    else
    {
        std::cout
            << "Sonuclar farkli. Kenar davranisini kontrol et.\n";
    }


    cv::imshow("Salt and Pepper - Orijinal", image);
    cv::imshow("OpenCV Median 5x5", cpuResult);
    cv::imshow("CUDA Median 5x5", gpuResult);

    cv::waitKey(0);
    cv::destroyAllWindows();


    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}