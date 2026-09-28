#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Gorev04/OpenCV_CUDA_Test";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout << "Goruntu okunamadi!" << std::endl;
                continue;
            }

            cv::resize(
                image,
                resizedImage,
                cv::Size(1024, 768)
            );

            cv::cvtColor(
                resizedImage,
                grayImage,
                cv::COLOR_BGR2GRAY
            );

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

            int mainHist[256] = { 0 };

            int hist1[256] = { 0 };
            int hist2[256] = { 0 };
            int hist3[256] = { 0 };
            int hist4[256] = { 0 };

    
            for (int y = 0; y < grayImage.rows; y++)
            {
                for (int x = 0; x < grayImage.cols; x++)
                {
                    int pixel = grayImage.at<uchar>(y, x);

                    mainHist[pixel]++;
                }
            }

            for (int y = 0; y < part1.rows; y++)
            {
                for (int x = 0; x < part1.cols; x++)
                {
                    int pixel = part1.at<uchar>(y, x);

                    hist1[pixel]++;
                }
            }

       
            for (int y = 0; y < part2.rows; y++)
            {
                for (int x = 0; x < part2.cols; x++)
                {
                    int pixel = part2.at<uchar>(y, x);

                    hist2[pixel]++;
                }
            }

           
            for (int y = 0; y < part3.rows; y++)
            {
                for (int x = 0; x < part3.cols; x++)
                {
                    int pixel = part3.at<uchar>(y, x);

                    hist3[pixel]++;
                }
            }

      
            for (int y = 0; y < part4.rows; y++)
            {
                for (int x = 0; x < part4.cols; x++)
                {
                    int pixel = part4.at<uchar>(y, x);

                    hist4[pixel]++;
                }
            }

          
            int combinedHist[256] = { 0 };

            for (int i = 0; i < 256; i++)
            {
                combinedHist[i] =
                    hist1[i] +
                    hist2[i] +
                    hist3[i] +
                    hist4[i];
            }

        
            bool same = true;

            for (int i = 0; i < 256; i++)
            {
                if (mainHist[i] != combinedHist[i])
                {
                    same = false;
                    break;
                }
            }

            std::cout << "\n============================\n";
            std::cout << "HISTOGRAM KONTROLU\n";
            std::cout << "============================\n";

            if (same)
            {
                std::cout
                    << "Histogramlar tamamen esit!"
                    << std::endl;
            }
            else
            {
                std::cout
                    << "Histogramlar esit degil!"
                    << std::endl;
            }

          
            long long mainTotal = 0;

            long long part1Total = 0;
            long long part2Total = 0;
            long long part3Total = 0;
            long long part4Total = 0;

            long long combinedTotal = 0;

            for (int i = 0; i < 256; i++)
            {
                mainTotal += mainHist[i];

                part1Total += hist1[i];
                part2Total += hist2[i];
                part3Total += hist3[i];
                part4Total += hist4[i];

                combinedTotal += combinedHist[i];
            }

            std::cout << "\n============================\n";
            std::cout << "PIXEL SAYILARI\n";
            std::cout << "============================\n";

            std::cout << "Ana goruntu        : "
                << mainTotal << std::endl;

            std::cout << "Part 1             : "
                << part1Total << std::endl;

            std::cout << "Part 2             : "
                << part2Total << std::endl;

            std::cout << "Part 3             : "
                << part3Total << std::endl;

            std::cout << "Part 4             : "
                << part4Total << std::endl;

            std::cout << "4 parca toplam     : "
                << part1Total +
                part2Total +
                part3Total +
                part4Total
                << std::endl;

            std::cout << "Birlesmis histogram: "
                << combinedTotal << std::endl;


          
            cv::Mat cpuResult = grayImage.clone();

            auto cpuStart =
                std::chrono::high_resolution_clock::now();

            for (int y = 0; y < grayImage.rows; y++)
            {
                for (int x = 0; x < grayImage.cols; x++)
                {
                    int pixel = grayImage.at<uchar>(y, x);

                    double newPixel = pixel * 0.75 + 20;

                    cpuResult.at<uchar>(y, x) =
                        cv::saturate_cast<uchar>(newPixel);
                }
            }

            auto cpuEnd =
                std::chrono::high_resolution_clock::now();

            double cpuTime =
                std::chrono::duration<double, std::milli>(
                    cpuEnd - cpuStart
                ).count();

            std::cout << "\n============================\n";
            std::cout << "CPU PERFORMANS\n";
            std::cout << "============================\n";

            std::cout << "CPU suresi: "
                << cpuTime
                << " ms"
                << std::endl;



            cv::imshow("Orijinal", resizedImage);

            cv::imshow("Gri", grayImage);

            cv::imshow(
                "CPU - 0.75x + 20",
                cpuResult
            );

            cv::waitKey(0);

            cv::destroyAllWindows();
        }
    }

    return 0;
}