#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

void sinyalOlustur(
    std::vector<double>& sinyal,
    double frekans,
    double genlik,
    double orneklemeFrekansi)
{
    for (int i = 0; i < sinyal.size(); i++)
    {
        double zaman = (double)i / orneklemeFrekansi;

        sinyal[i] =
            genlik * std::sin(
                2 * CV_PI * frekans * zaman
            );
    }
}

void gurultuEkle(
    std::vector<double>& sinyal,
    double gurultuGenligi)
{
    for (int i = 0; i < sinyal.size(); i++)
    {
        double rastgele =
            (double)rand() / RAND_MAX;

        double gurultu =
            (rastgele * 2.0 - 1.0)
            * gurultuGenligi;

        sinyal[i] += gurultu;
    }
}

std::vector<double> dftAnaliz(
    const std::vector<double>& sinyal)
{
    int N = sinyal.size();

    cv::Mat giris(N, 1, CV_64F);

    for (int i = 0; i < N; i++)
    {
        giris.at<double>(i, 0) = sinyal[i];
    }

    cv::Mat dftSonuc;

    cv::dft(
        giris,
        dftSonuc,
        cv::DFT_COMPLEX_OUTPUT
    );

    cv::Mat bilesenler[2];

    cv::split(
        dftSonuc,
        bilesenler
    );

    cv::Mat genlikler;

    cv::magnitude(
        bilesenler[0],
        bilesenler[1],
        genlikler
    );

    std::vector<double> harmonikler(N / 2);

    for (int i = 0; i < N / 2; i++)
    {
        harmonikler[i] =
            genlikler.at<double>(i, 0) / N;
    }

    return harmonikler;
}

std::vector<double> frekansFiltresi(
    const std::vector<double>& sinyal,
    double esikDegeri)
{
    int N = sinyal.size();

    cv::Mat giris(N, 1, CV_64F);

    for (int i = 0; i < N; i++)
    {
        giris.at<double>(i, 0) = sinyal[i];
    }

    cv::Mat dftSonuc;

    cv::dft(
        giris,
        dftSonuc,
        cv::DFT_COMPLEX_OUTPUT
    );

    for (int i = 0; i < N; i++)
    {
        cv::Vec2d& bilesen =
            dftSonuc.at<cv::Vec2d>(i, 0);

        double gercek = bilesen[0];
        double sanal = bilesen[1];

        double harmonik =
            std::sqrt(
                gercek * gercek +
                sanal * sanal
            );

        double normalizeHarmonik =
            harmonik / N;

        if (normalizeHarmonik < esikDegeri)
        {
            bilesen[0] = 0.0;
            bilesen[1] = 0.0;
        }
    }

    cv::Mat tersDft;

    cv::dft(
        dftSonuc,
        tersDft,
        cv::DFT_INVERSE |
        cv::DFT_REAL_OUTPUT |
        cv::DFT_SCALE
    );

    std::vector<double>
        filtrelenmisSinyal(N);

    for (int i = 0; i < N; i++)
    {
        filtrelenmisSinyal[i] =
            tersDft.at<double>(i, 0);
    }

    return filtrelenmisSinyal;
}

void sinyalGrafikCiz(
    const std::vector<double>& sinyal,
    const std::string& pencereAdi)
{
    int genislik = 1024;
    int yukseklik = 500;

    cv::Mat grafik(
        yukseklik,
        genislik,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    double minimum =
        *std::min_element(
            sinyal.begin(),
            sinyal.end()
        );

    double maksimum =
        *std::max_element(
            sinyal.begin(),
            sinyal.end()
        );

    double aralik =
        std::max(
            std::abs(minimum),
            std::abs(maksimum)
        );

    if (aralik == 0)
    {
        aralik = 1;
    }

    cv::line(
        grafik,
        cv::Point(0, yukseklik / 2),
        cv::Point(genislik - 1, yukseklik / 2),
        cv::Scalar(0, 0, 0),
        1
    );

    for (
        int i = 0;
        i < sinyal.size() - 1;
        i++
        )
    {
        int y1 =
            yukseklik / 2 -
            (int)(
                sinyal[i] / aralik * 220
                );

        int y2 =
            yukseklik / 2 -
            (int)(
                sinyal[i + 1] / aralik * 220
                );

        cv::line(
            grafik,
            cv::Point(i, y1),
            cv::Point(i + 1, y2),
            cv::Scalar(255, 0, 0),
            1
        );
    }

    cv::imshow(
        pencereAdi,
        grafik
    );
}

void harmonikGrafikCiz(
    const std::vector<double>& harmonikler,
    const std::string& pencereAdi)
{
    int genislik = 1024;
    int yukseklik = 500;

    cv::Mat grafik(
        yukseklik,
        genislik,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    double maksimum =
        *std::max_element(
            harmonikler.begin(),
            harmonikler.end()
        );

    if (maksimum == 0)
    {
        maksimum = 1;
    }

    cv::line(
        grafik,
        cv::Point(0, 450),
        cv::Point(genislik - 1, 450),
        cv::Scalar(0, 0, 0),
        1
    );

    for (int i = 0; i < harmonikler.size(); i++)
    {
        int x =
            (int)(
                (double)i /
                harmonikler.size() *
                (genislik - 1)
                );

        int cubukYuksekligi =
            (int)(
                harmonikler[i] /
                maksimum * 400
                );

        cv::line(
            grafik,
            cv::Point(x, 450),
            cv::Point(
                x,
                450 - cubukYuksekligi
            ),
            cv::Scalar(255, 0, 0),
            1
        );
    }

    cv::imshow(
        pencereAdi,
        grafik
    );
}

int main()
{
    srand(
        (unsigned int)time(0)
    );

    int N = 1024;

    double orneklemeFrekansi =
        1024.0;

    double esikDegeri = 0.5;

    std::vector<double>
        sinyal1(N);

    std::vector<double>
        sinyal2(N);

    std::vector<double>
        toplamSinyal(N);

    sinyalOlustur(
        sinyal1,
        50.0,
        12.0,
        orneklemeFrekansi
    );

    sinyalOlustur(
        sinyal2,
        30.0,
        5.0,
        orneklemeFrekansi
    );

    gurultuEkle(
        sinyal1,
        1.0
    );

    gurultuEkle(
        sinyal2,
        1.0
    );

    for (int i = 0; i < N; i++)
    {
        toplamSinyal[i] =
            sinyal1[i] +
            sinyal2[i];
    }

    std::vector<double>
        filtrelenmis1 =
        frekansFiltresi(
            sinyal1,
            esikDegeri
        );

    std::vector<double>
        filtrelenmis2 =
        frekansFiltresi(
            sinyal2,
            esikDegeri
        );

    std::vector<double>
        filtrelenmisToplam =
        frekansFiltresi(
            toplamSinyal,
            esikDegeri
        );

    std::vector<double>
        harmonik1 =
        dftAnaliz(sinyal1);

    std::vector<double>
        harmonik2 =
        dftAnaliz(sinyal2);

    std::vector<double>
        harmonikToplam =
        dftAnaliz(toplamSinyal);

    sinyalGrafikCiz(
        sinyal1,
        "Gurultulu 50 Hz Sinyal"
    );

    sinyalGrafikCiz(
        filtrelenmis1,
        "Filtrelenmis 50 Hz Sinyal"
    );

    sinyalGrafikCiz(
        sinyal2,
        "Gurultulu 30 Hz Sinyal"
    );

    sinyalGrafikCiz(
        filtrelenmis2,
        "Filtrelenmis 30 Hz Sinyal"
    );

    sinyalGrafikCiz(
        toplamSinyal,
        "Gurultulu Toplam Sinyal"
    );

    sinyalGrafikCiz(
        filtrelenmisToplam,
        "Filtrelenmis Toplam Sinyal"
    );

    harmonikGrafikCiz(
        harmonik1,
        "50 Hz DFT"
    );

    harmonikGrafikCiz(
        harmonik2,
        "30 Hz DFT"
    );

    harmonikGrafikCiz(
        harmonikToplam,
        "Toplam Sinyal DFT"
    );

    cv::waitKey(0);

    return 0;
}

