#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;
    cv::Mat opencvMedian;
    cv::Mat manualMedian;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Hafta3/UzamsalFiltreler5";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout << "Goruntu okunamadi: "
                    << entry.path() << std::endl;
                continue;
            }

            cv::resize(image, resizedImage, cv::Size(1024, 768));


            cv::cvtColor(
                resizedImage,
                grayImage,
                cv::COLOR_BGR2GRAY
            );

            cv::medianBlur(
                grayImage,
                opencvMedian,
                5
            );

            manualMedian = grayImage.clone();

            for (int y = 2; y < grayImage.rows - 2; y++)
            {
                for (int x = 2; x < grayImage.cols - 2; x++)
                {
                    int values[25];
                    int index = 0;

                    for (int ky = -2; ky <= 2; ky++)
                    {
                        for (int kx = -2; kx <= 2; kx++)
                        {
                            values[index] =
                                grayImage.at<uchar>(y + ky, x + kx);

                            index++;
                        }
                    }


                    std::sort(values, values + 25);

                    int median = values[12];

                    manualMedian.at<uchar>(y, x) =
                        static_cast<uchar>(median);
                }
            }


            cv::imwrite(
                (folderPath / "medyan_opencv.jpg").string(),
                opencvMedian
            );

            cv::imwrite(
                (folderPath / "medyan_manual.jpg").string(),
                manualMedian
            );

            cv::imshow("Orijinal", grayImage);
            cv::imshow("OpenCV Medyan 5x5", opencvMedian);
            cv::imshow("Manual Medyan 5x5", manualMedian);

            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return 0;
}