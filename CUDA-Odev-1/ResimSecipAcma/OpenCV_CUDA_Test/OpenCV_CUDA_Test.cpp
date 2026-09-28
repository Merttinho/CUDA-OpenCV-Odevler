#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;

    fs::path folderPath = "C:/Users/USER/Desktop/ResimSecipAcma/OpenCV_CUDA_Test";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                cout << "Goruntu okunamadi!" << endl;
            }
            else
            {
                cv::resize(image, resizedImage, cv::Size(1024, 768));

                cv::imshow("Mert", resizedImage);
                cv::waitKey(0);
            }
        }
    }

    cout << "Bitti." << endl;

    return 0;
}