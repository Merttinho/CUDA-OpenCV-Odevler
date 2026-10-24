#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main() {

    fs::path folder;
    folder = "C:/Users/USER/Desktop/GoruntuIsleme/Hafta5/Gorev05";
    cv::Mat grayImage;
    cv::Mat newImage;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".jpg") {

            cv::Mat image = cv::imread(entry.path().string());

            if (image.empty()) {
                std::cout << "Goruntu Okunamadi\n";
                continue;
            }

            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            cv::applyColorMap(grayImage, newImage, cv::COLORMAP_TURBO);


            cv::imshow("Gorseller", grayImage);
            cv::imshow("Renklendirilimis", newImage);
            cv::waitKey(0);
            
        }
    }

    return 0;
}

