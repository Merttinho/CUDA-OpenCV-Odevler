#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat ikiBoyutluDFT(const cv::Mat& image)
{
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);

    cv::Mat complexImage;

    cv::dft(
        imageFloat,
        complexImage,
        cv::DFT_COMPLEX_OUTPUT
    );

    return complexImage;
}

cv::Mat harmonikGoster(const cv::Mat& complexImage)
{
    cv::Mat bilesenler[2];

    cv::split(
        complexImage,
        bilesenler
    );

    cv::Mat magnitude;

    cv::magnitude(
        bilesenler[0],
        bilesenler[1],
        magnitude
    );

    magnitude += 1;

    cv::log(
        magnitude,
        magnitude
    );

    magnitude = magnitude(
        cv::Rect(
            0,
            0,
            magnitude.cols & -2,
            magnitude.rows & -2
        )
    );

    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;

    cv::Mat q0(
        magnitude,
        cv::Rect(0, 0, cx, cy)
    );

    cv::Mat q1(
        magnitude,
        cv::Rect(cx, 0, cx, cy)
    );

    cv::Mat q2(
        magnitude,
        cv::Rect(0, cy, cx, cy)
    );

    cv::Mat q3(
        magnitude,
        cv::Rect(cx, cy, cx, cy)
    );

    cv::Mat temp;

    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);

    cv::normalize(
        magnitude,
        magnitude,
        0,
        255,
        cv::NORM_MINMAX
    );

    magnitude.convertTo(
        magnitude,
        CV_8U
    );

    return magnitude;
}

void gaussGurultuEkle(
    const cv::Mat& image,
    cv::Mat& sonuc,
    double ortalama,
    double sapma)
{
    cv::Mat gurultu(
        image.size(),
        CV_32F
    );

    cv::randn(
        gurultu,
        ortalama,
        sapma
    );

    cv::Mat imageFloat;

    image.convertTo(
        imageFloat,
        CV_32F
    );

    cv::Mat gurultulu =
        imageFloat + gurultu;

    gurultulu.convertTo(
        sonuc,
        CV_8U
    );
}

void saltPepperGurultuEkle(
    const cv::Mat& image,
    cv::Mat& sonuc,
    double oran)
{
    sonuc = image.clone();

    int pikselSayisi =
        image.rows * image.cols;

    int gurultuluPikselSayisi =
        pikselSayisi * oran;

    for (
        int i = 0;
        i < gurultuluPikselSayisi;
        i++
        )
    {
        int y =
            rand() % image.rows;

        int x =
            rand() % image.cols;

        int deger =
            rand() % 2;

        if (deger == 0)
        {
            sonuc.at<uchar>(y, x) = 0;
        }
        else
        {
            sonuc.at<uchar>(y, x) = 255;
        }
    }
}

int main()
{
    srand(
        (unsigned int)time(0)
    );

    cv::Mat image = cv::imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta7/Gorev07/resim.jpeg",
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cout
            << "Goruntu okunamadi"
            << std::endl;

        return -1;
    }

    cv::Mat gaussGurultulu;

    gaussGurultuEkle(
        image,
        gaussGurultulu,
        0,
        25
    );

    cv::Mat saltPepperGurultulu;

    saltPepperGurultuEkle(
        image,
        saltPepperGurultulu,
        0.05
    );

    cv::Mat orijinalDFT =
        ikiBoyutluDFT(image);

    cv::Mat gaussDFT =
        ikiBoyutluDFT(
            gaussGurultulu
        );

    cv::Mat saltPepperDFT =
        ikiBoyutluDFT(
            saltPepperGurultulu
        );

    cv::Mat orijinalHarmonik =
        harmonikGoster(
            orijinalDFT
        );

    cv::Mat gaussHarmonik =
        harmonikGoster(
            gaussDFT
        );

    cv::Mat saltPepperHarmonik =
        harmonikGoster(
            saltPepperDFT
        );

    cv::imshow(
        "Orijinal",
        image
    );

    cv::imshow(
        "Gauss Gurultulu",
        gaussGurultulu
    );

    cv::imshow(
        "Salt Pepper Gurultulu",
        saltPepperGurultulu
    );

    cv::imshow(
        "Orijinal Harmonik",
        orijinalHarmonik
    );

    cv::imshow(
        "Gauss Harmonik",
        gaussHarmonik
    );

    cv::imshow(
        "Salt Pepper Harmonik",
        saltPepperHarmonik
    );

    cv::waitKey(0);

    return 0;
}
