#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;
    cv::Mat equalizedImage;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Hafta3/UzamsalFiltreler2";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
         
            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout << "Goruntu Okunamadi: "
                    << entry.path() << std::endl;
                continue;
            }

           
            cv::resize(image, resizedImage, cv::Size(1024, 768));

          
            cv::cvtColor(
                resizedImage,
                grayImage,
                cv::COLOR_BGR2GRAY
            );

         
            equalizedImage = grayImage.clone();

          
            int hist[256] = { 0 };

            for (int y = 0; y < grayImage.rows; y++)
            {
                for (int x = 0; x < grayImage.cols; x++)
                {
                    int pixel = grayImage.at<uchar>(y, x);

                    hist[pixel]++;
                }
            }

       
            int cdf[256] = { 0 };

            cdf[0] = hist[0];

            for (int i = 1; i < 256; i++)
            {
                cdf[i] = cdf[i - 1] + hist[i];
            }


            int cdfMin = 0;

            for (int i = 0; i < 256; i++)
            {
                if (cdf[i] != 0)
                {
                    cdfMin = cdf[i];
                    break;
                }
            }


            int totalPixels =
                grayImage.rows * grayImage.cols;

            for (int y = 0; y < grayImage.rows; y++)
            {
                for (int x = 0; x < grayImage.cols; x++)
                {
                    int pixel =
                        grayImage.at<uchar>(y, x);

                    int newValue = static_cast<int>(
                        ((cdf[pixel] - cdfMin) * 255.0) /
                        (totalPixels - cdfMin)
                        );

               
                    if (newValue < 0)
                        newValue = 0;

                    if (newValue > 255)
                        newValue = 255;

                    equalizedImage.at<uchar>(y, x) =
                        static_cast<uchar>(newValue);
                }
            }

     
            std::string outputPath =
                (folderPath / "esitlenmis.jpg").string();

            cv::imwrite(
                outputPath,
                equalizedImage
            );

            

            cv::imshow(
                "Orijinal",
                grayImage
            );

            cv::imshow(
                "Histogram Esitlenmis",
                equalizedImage
            );

            cv::waitKey(0);

            cv::destroyAllWindows();
        }
    }

    return 0;
}