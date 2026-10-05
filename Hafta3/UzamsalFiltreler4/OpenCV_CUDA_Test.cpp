#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;
    cv::Mat filteredImage;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Hafta3/UzamsalFiltreler4";

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png" ||
            entry.path().filename() != "ortalama_filtre.jpg")
        {

            image = cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout << "Goruntu okunamadi: "
                    << entry.path() << std::endl;
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


            filteredImage = grayImage.clone();



            for (int y = 1; y < grayImage.rows - 1; y++)
            {
                for (int x = 1; x < grayImage.cols - 1; x++)
                {
                    int sum = 0;

  
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            sum += grayImage.at<uchar>(
                                y + ky,
                                x + kx
                            );
                        }
                    }

                    int average = sum / 9;

                    filteredImage.at<uchar>(y, x) =
                        static_cast<uchar>(average);
                }
            }


            cv::imwrite(
                (folderPath / "ortalama_filtre.jpg").string(),
                filteredImage
            );


            cv::imshow(
                "Gurultulu Goruntu",
                grayImage
            );

            cv::imshow(
                "3x3 Ortalama Filtre",
                filteredImage
            );

            cv::waitKey(0);
        }
    }

    return 0;
}