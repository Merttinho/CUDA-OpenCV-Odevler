#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;

    fs::path folderPath = "C:/Users/USER/Desktop/Gorev01/OpenCV_CUDA_Test";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            image = cv::imread(entry.path().string());

            std::cout << entry.path().extension() << std::endl;

            if (image.empty())
            {
                std::cout << "Goruntu okunamadi!" << std::endl;
            }
            else
            {
                cv::resize(image, resizedImage, cv::Size(1024, 768));

                cv::cvtColor(
                    resizedImage,
                    grayImage,
                    cv::COLOR_BGR2GRAY
                );

                cv::Mat dividedImage = grayImage.clone();
                cv::Mat multipliedImage = grayImage.clone();

               
                for (int y = 0; y < grayImage.rows; y++)
                {
                    for (int x = 0; x < grayImage.cols; x++)
                    {
                 
                        int pixel = grayImage.at<uchar>(y, x);

                  
                        dividedImage.at<uchar>(y, x) = pixel / 4;

             
                        int value = pixel * 4;

                 
                        if (value > 255)
                        {
                            value = 255;
                        }

                        multipliedImage.at<uchar>(y, x) = value;
                    }
                }

                cv::imshow("Orijinal", resizedImage);
                cv::waitKey(0);
                cv::imshow("Gri", grayImage);
                cv::waitKey(0);
                cv::imshow("4'e Bolunmus", dividedImage);
                cv::waitKey(0);
                cv::imshow("4 ile Carpilmis", multipliedImage);

                cv::waitKey(0);
            }
        }
    }

    return 0;
}