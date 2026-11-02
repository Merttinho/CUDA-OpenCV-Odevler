#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>

cv::Mat ikiBoyutluDFT(const cv::Mat& image)
{
    cv::Mat imageFloat;

    image.convertTo(
        imageFloat,
        CV_32F
    );

    cv::Mat complexImage;

    cv::dft(
        imageFloat,
        complexImage,
        cv::DFT_COMPLEX_OUTPUT
    );

    return complexImage;
}

cv::Mat tersDFT(const cv::Mat& complexImage)
{
    cv::Mat sonuc;

    cv::dft(
        complexImage,
        sonuc,
        cv::DFT_INVERSE |
        cv::DFT_REAL_OUTPUT |
        cv::DFT_SCALE
    );

    sonuc.convertTo(
        sonuc,
        CV_8U
    );

    return sonuc;
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
        (int)(pikselSayisi * oran);

    for (int i = 0; i < gurultuluPikselSayisi; i++)
    {
        int y = rand() % image.rows;
        int x = rand() % image.cols;

        if (rand() % 2 == 0)
        {
            sonuc.at<uchar>(y, x) = 0;
        }
        else
        {
            sonuc.at<uchar>(y, x) = 255;
        }
    }
}

double znccHesapla(
    const cv::Mat& orijinal,
    const cv::Mat& karsilastirilan)
{
    cv::Mat a;
    cv::Mat b;

    orijinal.convertTo(a, CV_64F);
    karsilastirilan.convertTo(b, CV_64F);

    cv::Scalar ortalamaA;
    cv::Scalar stdA;
    cv::Scalar ortalamaB;
    cv::Scalar stdB;

    cv::meanStdDev(a, ortalamaA, stdA);
    cv::meanStdDev(b, ortalamaB, stdB);

    double pay = 0.0;
    double paydaA = 0.0;
    double paydaB = 0.0;

    for (int y = 0; y < a.rows; y++)
    {
        for (int x = 0; x < a.cols; x++)
        {
            double farkA =
                a.at<double>(y, x) - ortalamaA[0];

            double farkB =
                b.at<double>(y, x) - ortalamaB[0];

            pay += farkA * farkB;

            paydaA += farkA * farkA;
            paydaB += farkB * farkB;
        }
    }

    double payda =
        std::sqrt(paydaA * paydaB);

    if (payda == 0.0)
    {
        return 0.0;
    }

    return pay / payda;
}

double mseHesapla(
    const cv::Mat& orijinal,
    const cv::Mat& karsilastirilan)
{
    cv::Mat a;
    cv::Mat b;

    orijinal.convertTo(a, CV_64F);
    karsilastirilan.convertTo(b, CV_64F);

    cv::Mat fark = a - b;

    cv::Mat kare;

    cv::multiply(
        fark,
        fark,
        kare
    );

    return cv::sum(kare)[0] /
        (orijinal.rows * orijinal.cols);
}

double psnrHesapla(double mse)
{
    if (mse == 0.0)
    {
        return INFINITY;
    }

    return 10.0 * std::log10(
        (255.0 * 255.0) / mse
    );
}

int main()
{
    srand((unsigned int)time(0));

    cv::Mat image = cv::imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta7/Gorev08 - Kopya/resim.jpeg",
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cout << "Goruntu okunamadi!" << std::endl;
        return -1;
    }

    cv::Mat gaussGurultulu;
    cv::Mat saltPepperGurultulu;

    gaussGurultuEkle(
        image,
        gaussGurultulu,
        0,
        25
    );

    saltPepperGurultuEkle(
        image,
        saltPepperGurultulu,
        0.05
    );

    cv::Mat orijinalDFT =
        ikiBoyutluDFT(image);

    cv::Mat gaussDFT =
        ikiBoyutluDFT(gaussGurultulu);

    cv::Mat saltPepperDFT =
        ikiBoyutluDFT(saltPepperGurultulu);

    cv::Mat orijinalYeniden =
        tersDFT(orijinalDFT);

    cv::Mat gaussYeniden =
        tersDFT(gaussDFT);

    cv::Mat saltPepperYeniden =
        tersDFT(saltPepperDFT);

    double znccOrijinal =
        znccHesapla(image, orijinalYeniden);

    double znccGauss =
        znccHesapla(image, gaussYeniden);

    double znccSaltPepper =
        znccHesapla(image, saltPepperYeniden);

    double mseOrijinal =
        mseHesapla(image, orijinalYeniden);

    double mseGauss =
        mseHesapla(image, gaussYeniden);

    double mseSaltPepper =
        mseHesapla(image, saltPepperYeniden);

    std::cout << "\nORIJINAL GORUNTU" << std::endl;
    std::cout << "ZNCC: " << znccOrijinal << std::endl;
    std::cout << "MSE: " << mseOrijinal << std::endl;
    std::cout << "PSNR: " << psnrHesapla(mseOrijinal) << " dB" << std::endl;

    std::cout << "\nGAUSS GURULTULU GORUNTU" << std::endl;
    std::cout << "ZNCC: " << znccGauss << std::endl;
    std::cout << "MSE: " << mseGauss << std::endl;
    std::cout << "PSNR: " << psnrHesapla(mseGauss) << " dB" << std::endl;

    std::cout << "\nSALT AND PEPPER GURULTULU GORUNTU" << std::endl;
    std::cout << "ZNCC: " << znccSaltPepper << std::endl;
    std::cout << "MSE: " << mseSaltPepper << std::endl;
    std::cout << "PSNR: " << psnrHesapla(mseSaltPepper) << " dB" << std::endl;

    cv::imshow("Orijinal", image);
    cv::imshow("Gauss Gurultulu", gaussGurultulu);
    cv::imshow("Gauss Ters DFT", gaussYeniden);

    cv::imshow("Salt Pepper Gurultulu", saltPepperGurultulu);
    cv::imshow("Salt Pepper Ters DFT", saltPepperYeniden);

    cv::imshow("Orijinal Ters DFT", orijinalYeniden);

    cv::waitKey(0);

    return 0;
}
