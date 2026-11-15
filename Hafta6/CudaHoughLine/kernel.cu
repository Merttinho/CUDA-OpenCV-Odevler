#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define PI 3.14159265358979323846

__global__ void houghCUDA(
    const uchar* edges,
    int width,
    int height,
    int* accumulator,
    int rhoMax,
    int thetaCount,
    const float* cosTable,
    const float* sinTable
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int toplamPiksel = width * height;

    if (index >= toplamPiksel)
        return;

    int x = index % width;
    int y = index / width;

    if (edges[index] == 0)
        return;

    for (int theta = 0; theta < thetaCount; theta++) {

        float rho = x * cosTable[theta] + y * sinTable[theta];

        int rhoIndex = roundf(rho) + rhoMax;

        if (rhoIndex >= 0 && rhoIndex < 2 * rhoMax + 1) {

            int accumulatorIndex = theta * (2 * rhoMax + 1) + rhoIndex;

            atomicAdd(&accumulator[accumulatorIndex], 1);
        }
    }
}

Mat houghCPU(const Mat& edges, int thetaCount, int rhoMax) {

    int rhoSize = 2 * rhoMax + 1;

    Mat accumulator = Mat::zeros(thetaCount, rhoSize, CV_32S);

    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {

            if (edges.at<uchar>(y, x) == 0)
                continue;

            for (int theta = 0; theta < thetaCount; theta++) {

                float angle = theta * PI / 180.0;

                float rho = x * cos(angle) + y * sin(angle);

                int rhoIndex = round(rho) + rhoMax;

                if (rhoIndex >= 0 && rhoIndex < rhoSize) {
                    accumulator.at<int>(theta, rhoIndex)++;
                }
            }
        }
    }

    return accumulator;
}

int main() {

    Mat image = imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta6/CUDA/CudaHoughLine/NoıseandLınes.jpeg",
        IMREAD_GRAYSCALE
    );

    if (image.empty()) {
        cout << "Goruntu yuklenemedi!" << endl;
        return -1;
    }

    Mat edges;

    Canny(image, edges, 50, 150);

    int width = edges.cols;
    int height = edges.rows;

    int thetaCount = 180;

    int rhoMax = static_cast<int>(
        sqrt(width * width + height * height)
        );

    int rhoSize = 2 * rhoMax + 1;

    vector<float> cosTable(thetaCount);
    vector<float> sinTable(thetaCount);

    for (int i = 0; i < thetaCount; i++) {

        float angle = i * PI / 180.0;

        cosTable[i] = cos(angle);
        sinTable[i] = sin(angle);
    }

    auto cpuStart = chrono::high_resolution_clock::now();

    Mat cpuAccumulator = houghCPU(edges, thetaCount, rhoMax);

    auto cpuEnd = chrono::high_resolution_clock::now();

    double cpuTime = chrono::duration<double, milli>(
        cpuEnd - cpuStart
    ).count();

    uchar* d_edges;
    int* d_accumulator;
    float* d_cosTable;
    float* d_sinTable;

    int* h_accumulator = new int[thetaCount * rhoSize];

    cudaMalloc(&d_edges, width * height * sizeof(uchar));

    cudaMalloc(
        &d_accumulator,
        thetaCount * rhoSize * sizeof(int)
    );

    cudaMalloc(&d_cosTable, thetaCount * sizeof(float));
    cudaMalloc(&d_sinTable, thetaCount * sizeof(float));

    cudaMemcpy(
        d_edges,
        edges.data,
        width * height * sizeof(uchar),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_cosTable,
        cosTable.data(),
        thetaCount * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_sinTable,
        sinTable.data(),
        thetaCount * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemset(
        d_accumulator,
        0,
        thetaCount * rhoSize * sizeof(int)
    );

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (width * height + threadsPerBlock - 1) / threadsPerBlock;

    auto gpuStart = chrono::high_resolution_clock::now();

    houghCUDA << <blocksPerGrid, threadsPerBlock >> > (
        d_edges,
        width,
        height,
        d_accumulator,
        rhoMax,
        thetaCount,
        d_cosTable,
        d_sinTable
        );

    cudaDeviceSynchronize();

    auto gpuEnd = chrono::high_resolution_clock::now();

    cudaMemcpy(
        h_accumulator,
        d_accumulator,
        thetaCount * rhoSize * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    double gpuTime = chrono::duration<double, milli>(
        gpuEnd - gpuStart
    ).count();

    Mat gpuAccumulator(thetaCount, rhoSize, CV_32S, h_accumulator);

    Mat cpuDisplay;
    Mat gpuDisplay;

    normalize(cpuAccumulator, cpuDisplay, 0, 255, NORM_MINMAX);
    normalize(gpuAccumulator, gpuDisplay, 0, 255, NORM_MINMAX);

    cpuDisplay.convertTo(cpuDisplay, CV_8U);
    gpuDisplay.convertTo(gpuDisplay, CV_8U);

    cout << "CPU Hough suresi: " << cpuTime << " ms" << endl;
    cout << "CUDA Hough suresi: " << gpuTime << " ms" << endl;

    if (gpuTime > 0) {
        cout << "Hizlanma: " << cpuTime / gpuTime << "x" << endl;
    }

    Mat cpuResult = image.clone();
    Mat gpuResult = image.clone();

    HoughLinesP(edges, vector<Vec4i>(), 1, PI / 180, 100);

    imshow("Orijinal", image);
    imshow("Kenarlar", edges);
    imshow("CPU Hough Accumulator", cpuDisplay);
    imshow("CUDA Hough Accumulator", gpuDisplay);

    waitKey(0);

    cudaFree(d_edges);
    cudaFree(d_accumulator);
    cudaFree(d_cosTable);
    cudaFree(d_sinTable);

    delete[] h_accumulator;

    return 0;
}