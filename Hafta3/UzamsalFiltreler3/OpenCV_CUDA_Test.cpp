#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;


// ============================================================
// MANUEL CLAHE
// ============================================================

cv::Mat manualCLAHE(
    const cv::Mat& grayImage,
    double clipLimit = 40.0,
    int tileRows = 8,
    int tileCols = 8)
{
    int rows = grayImage.rows;
    int cols = grayImage.cols;

    // Her bölgenin LUT'u:
    // [tileRow][tileCol][0-255]
    std::vector<std::vector<std::vector<unsigned char>>> LUT(
        tileRows,
        std::vector<std::vector<unsigned char>>(
            tileCols,
            std::vector<unsigned char>(256, 0)
        )
    );

    // --------------------------------------------------------
    // 1. GÖRÜNTÜYÜ BÖLGELERE AYIR
    // --------------------------------------------------------

    for (int ty = 0; ty < tileRows; ty++)
    {
        for (int tx = 0; tx < tileCols; tx++)
        {
            int yStart = ty * rows / tileRows;
            int yEnd = (ty + 1) * rows / tileRows;

            int xStart = tx * cols / tileCols;
            int xEnd = (tx + 1) * cols / tileCols;

            int tileWidth = xEnd - xStart;
            int tileHeight = yEnd - yStart;

            int tileArea = tileWidth * tileHeight;

            // ------------------------------------------------
            // 2. BU BÖLGENİN HISTOGRAMINI ÇIKAR
            // ------------------------------------------------

            int hist[256] = { 0 };

            for (int y = yStart; y < yEnd; y++)
            {
                for (int x = xStart; x < xEnd; x++)
                {
                    int pixel = grayImage.at<uchar>(y, x);

                    hist[pixel]++;
                }
            }

            // ------------------------------------------------
            // 3. CLIP LIMIT HESAPLA
            // ------------------------------------------------

            int limit = static_cast<int>(
                clipLimit * tileArea / 256.0
                );

            if (limit < 1)
                limit = 1;

            // ------------------------------------------------
            // 4. HISTOGRAMI CLIP ET
            // ------------------------------------------------

            int excess = 0;

            for (int i = 0; i < 256; i++)
            {
                if (hist[i] > limit)
                {
                    excess += hist[i] - limit;
                    hist[i] = limit;
                }
            }

            // ------------------------------------------------
            // 5. EXCESS DEĞERLERİ TÜM HISTOGRAMA DAĞIT
            // ------------------------------------------------

            int addition = excess / 256;
            int remainder = excess % 256;

            for (int i = 0; i < 256; i++)
            {
                hist[i] += addition;
            }

            // Kalan değerleri dağıt
            for (int i = 0; i < remainder; i++)
            {
                hist[i]++;
            }

            // ------------------------------------------------
            // 6. CDF HESAPLA
            // ------------------------------------------------

            int cdf[256] = { 0 };

            cdf[0] = hist[0];

            for (int i = 1; i < 256; i++)
            {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            // İlk sıfır olmayan CDF
            int cdfMin = 0;

            for (int i = 0; i < 256; i++)
            {
                if (cdf[i] != 0)
                {
                    cdfMin = cdf[i];
                    break;
                }
            }

            // ------------------------------------------------
            // 7. LUT OLUŞTUR
            // ------------------------------------------------

            for (int i = 0; i < 256; i++)
            {
                if (tileArea == cdfMin)
                {
                    LUT[ty][tx][i] = static_cast<unsigned char>(i);
                }
                else
                {
                    int newValue = static_cast<int>(
                        ((cdf[i] - cdfMin) * 255.0) /
                        (tileArea - cdfMin)
                        );

                    if (newValue < 0)
                        newValue = 0;

                    if (newValue > 255)
                        newValue = 255;

                    LUT[ty][tx][i] =
                        static_cast<unsigned char>(newValue);
                }
            }
        }
    }

    // ========================================================
    // 8. BİLİNEAR INTERPOLATION İLE SONUCU OLUŞTUR
    // ========================================================

    cv::Mat result = grayImage.clone();

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            // Görüntüdeki konumu 0-7 aralığına dönüştür
            double gy =
                ((double)y / rows) * tileRows - 0.5;

            double gx =
                ((double)x / cols) * tileCols - 0.5;

            int y1 = static_cast<int>(floor(gy));
            int x1 = static_cast<int>(floor(gx));

            double fy = gy - y1;
            double fx = gx - x1;

            int y2 = y1 + 1;
            int x2 = x1 + 1;

            // Sınırları kontrol et
            if (y1 < 0)
            {
                y1 = 0;
                y2 = 0;
                fy = 0;
            }

            if (x1 < 0)
            {
                x1 = 0;
                x2 = 0;
                fx = 0;
            }

            if (y2 >= tileRows)
            {
                y2 = tileRows - 1;
                y1 = y2;
                fy = 0;
            }

            if (x2 >= tileCols)
            {
                x2 = tileCols - 1;
                x1 = x2;
                fx = 0;
            }

            int pixel = grayImage.at<uchar>(y, x);

            // Dört komşu bölgenin sonucu
            double topLeft =
                LUT[y1][x1][pixel];

            double topRight =
                LUT[y1][x2][pixel];

            double bottomLeft =
                LUT[y2][x1][pixel];

            double bottomRight =
                LUT[y2][x2][pixel];

            // Yatay interpolasyon
            double top =
                topLeft * (1.0 - fx) +
                topRight * fx;

            double bottom =
                bottomLeft * (1.0 - fx) +
                bottomRight * fx;

            // Dikey interpolasyon
            double value =
                top * (1.0 - fy) +
                bottom * fy;

            if (value < 0)
                value = 0;

            if (value > 255)
                value = 255;

            result.at<uchar>(y, x) =
                static_cast<unsigned char>(value);
        }
    }

    return result;
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    cv::Mat image;
    cv::Mat resizedImage;
    cv::Mat grayImage;

    fs::path folderPath =
        "C:/Users/USER/Desktop/Hafta3/UzamsalFiltreler2 - Kopya";

    for (const auto& entry :
        fs::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png")
        {
            // ------------------------------------------------
            // 1. GÖRÜNTÜYÜ AÇ
            // ------------------------------------------------

            image =
                cv::imread(entry.path().string());

            if (image.empty())
            {
                std::cout
                    << "Goruntu okunamadi: "
                    << entry.path()
                    << std::endl;

                continue;
            }

            // ------------------------------------------------
            // 2. BOYUTLANDIR
            // ------------------------------------------------

            cv::resize(
                image,
                resizedImage,
                cv::Size(1024, 768)
            );

            // ------------------------------------------------
            // 3. TEK KANALA ÇEVİR
            // ------------------------------------------------

            cv::cvtColor(
                resizedImage,
                grayImage,
                cv::COLOR_BGR2GRAY
            );


            // =================================================
            // 4. OPENCV HAZIR CLAHE
            // =================================================

            cv::Ptr<cv::CLAHE> clahe =
                cv::createCLAHE(
                    40.0,
                    cv::Size(8, 8)
                );

            cv::Mat opencvCLAHE;

            clahe->apply(
                grayImage,
                opencvCLAHE
            );


            // =================================================
            // 5. KENDİ CLAHE'MİZ
            // =================================================

            cv::Mat myCLAHE =
                manualCLAHE(
                    grayImage,
                    40.0,
                    8,
                    8
                );


            // =================================================
            // 6. SONUÇLARI KAYDET
            // =================================================

            cv::imwrite(
                (folderPath / "CLAHE_OpenCV.jpg").string(),
                opencvCLAHE
            );

            cv::imwrite(
                (folderPath / "CLAHE_Manual.jpg").string(),
                myCLAHE
            );


            // =================================================
            // 7. SONUÇLARI GÖSTER
            // =================================================

            cv::imshow(
                "Orijinal",
                grayImage
            );

            cv::imshow(
                "OpenCV CLAHE",
                opencvCLAHE
            );

            cv::imshow(
                "Manual CLAHE",
                myCLAHE
            );

            cv::waitKey(0);

            cv::destroyAllWindows();
        }
    }

    return 0;
}