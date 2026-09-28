#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;

    fs::path folderPath = "C:/Users/USER/Desktop/Gorev02/OpenCV_CUDA_Test";

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
                continue;
            }


            cv::resize(image, resizedImage, cv::Size(1024, 768));


            cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);


            int halfWidth = grayImage.cols / 2;
            int halfHeight = grayImage.rows / 2;


            cv::Mat part1 = grayImage(
                cv::Rect(0, 0, halfWidth, halfHeight)
            );


            cv::Mat part2 = grayImage(
                cv::Rect(halfWidth, 0, halfWidth, halfHeight)
            );

            cv::Mat part3 = grayImage(
                cv::Rect(0, halfHeight, halfWidth, halfHeight)
            );


            cv::Mat part4 = grayImage(
                cv::Rect(halfWidth, halfHeight, halfWidth, halfHeight)
            );



            cv::Mat part1Div = part1.clone();
            cv::Mat part2Div = part2.clone();
            cv::Mat part3Div = part3.clone();
            cv::Mat part4Div = part4.clone();

            for (int y = 0; y < part1.rows; y++)
            {
                for (int x = 0; x < part1.cols; x++)
                {
                    part1Div.at<uchar>(y, x) =
                        part1.at<uchar>(y, x) / 4;

                    part2Div.at<uchar>(y, x) =
                        part2.at<uchar>(y, x) / 4;

                    part3Div.at<uchar>(y, x) =
                        part3.at<uchar>(y, x) / 4;

                    part4Div.at<uchar>(y, x) =
                        part4.at<uchar>(y, x) / 4;
                }
            }


            cv::Mat part1Mul = part1.clone();
            cv::Mat part2Mul = part2.clone();
            cv::Mat part3Mul = part3.clone();
            cv::Mat part4Mul = part4.clone();

            for (int y = 0; y < part1.rows; y++)
            {
                for (int x = 0; x < part1.cols; x++)
                {
                    part1Mul.at<uchar>(y, x) =
                        cv::saturate_cast<uchar>(
                            part1.at<uchar>(y, x) * 4
                        );

                    part2Mul.at<uchar>(y, x) =
                        cv::saturate_cast<uchar>(
                            part2.at<uchar>(y, x) * 4
                        );

                    part3Mul.at<uchar>(y, x) =
                        cv::saturate_cast<uchar>(
                            part3.at<uchar>(y, x) * 4
                        );

                    part4Mul.at<uchar>(y, x) =
                        cv::saturate_cast<uchar>(
                            part4.at<uchar>(y, x) * 4
                        );
                }
            }

            cv::imshow("Orijinal", resizedImage);
            cv::waitKey(0);
            cv::imshow("Gri", grayImage);
            cv::waitKey(0);
        }
    }
}