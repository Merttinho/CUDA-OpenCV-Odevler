#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Hafta3/UzamsalFiltreler";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout << "Goruntu Okunamadi" << std::endl;
                continue;
            }

            cv::resize(image, resizedImage, cv::Size(1024, 768));

            cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

        
            int minValue = 255;
            int maxValue = 0;

            for (int y = 0; y < grayImage.rows; y++)
            {
                for (int x = 0; x < grayImage.cols; x++)
                {
                    int pixel = grayImage.at<uchar>(y, x);

                    if (pixel < minValue)
                    {
                        minValue = pixel;
                    }

                    if (pixel > maxValue)
                    {
                        maxValue = pixel;
                    }
                }
            }

            std::cout << "Minimum: " << minValue << std::endl;
            std::cout << "Maximum: " << maxValue << std::endl;

   
            cv::Mat contrastImage = grayImage.clone();

            if (maxValue != minValue)
            {
                for (int y = 0; y < grayImage.rows; y++)
                {
                    for (int x = 0; x < grayImage.cols; x++)
                    {
                        int pixel = grayImage.at<uchar>(y, x);

                        int newPixel =
                            (pixel - minValue) * 255 / (maxValue - minValue);

                        contrastImage.at<uchar>(y, x) = newPixel;
                    }
                }
            }

    
            cv::imshow("Orijinal", grayImage);
            cv::waitKey(0);

            cv::imshow("Kontrast", contrastImage);
            cv::waitKey(0);

            cv::destroyAllWindows();
        }
    }

    return 0;
}