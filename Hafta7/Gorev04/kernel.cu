
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

void sinyalOlustur(std::vector<double>& sinyal,
    double frekans,
    double genlik,
    double orneklemeFrekansi) {

    for (int i = 0; i < sinyal.size(); i++) {

        double zaman = (double)i / orneklemeFrekansi;

        sinyal[i] = genlik * std::sin(2 * CV_PI * frekans * zaman);
    }
}

void gurultuEkle(std::vector<double>& sinyal, double gurultuGenligi) {

    for (int i = 0; i < sinyal.size(); i++) {

        double rastgele = (double)rand() / RAND_MAX;

        double gurultu = (rastgele * 2.0 - 1.0) * gurultuGenligi;

        sinyal[i] += gurultu;
    }
}

std::vector<double> dftAnaliz(const std::vector<double>& sinyal) {

    int N = sinyal.size();

    cv::Mat giris(N, 1, CV_64F);

    for (int i = 0; i < N; i++) {
        giris.at<double>(i, 0) = sinyal[i];
    }

    cv::Mat dftSonuc;

    cv::dft(giris, dftSonuc, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat bilesenler[2];

    cv::split(dftSonuc, bilesenler);

    cv::Mat genlikler;

    cv::magnitude(bilesenler[0], bilesenler[1], genlikler);

    std::vector<double> harmonikler(N / 2 + 1);

    for (int i = 0; i <= N / 2; i++) {

        double deger = genlikler.at<double>(i, 0) / N;

        if (i != 0 && i != N / 2) {
            deger *= 2.0;
        }

        harmonikler[i] = deger;
    }

    return harmonikler;
}

void sinyalGrafikCiz(const std::vector<double>& sinyal,
    const std::string& pencereAdi) {

    int genislik = 1024;
    int yukseklik = 500;

    cv::Mat grafik(yukseklik, genislik, CV_8UC3,
        cv::Scalar(255, 255, 255));

    double minimum = *std::min_element(sinyal.begin(), sinyal.end());
    double maksimum = *std::max_element(sinyal.begin(), sinyal.end());

    double aralik = std::max(std::abs(minimum), std::abs(maksimum));

    if (aralik == 0) aralik = 1;

    cv::line(grafik,
        cv::Point(0, yukseklik / 2),
        cv::Point(genislik - 1, yukseklik / 2),
        cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i < sinyal.size() - 1; i++) {

        int x1 = i;
        int x2 = i + 1;

        int y1 = yukseklik / 2 -
            (int)(sinyal[i] / aralik * 220);

        int y2 = yukseklik / 2 -
            (int)(sinyal[i + 1] / aralik * 220);

        cv::line(grafik,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow(pencereAdi, grafik);
}

void dftGrafikCiz(const std::vector<double>& harmonikler,
    const std::string& pencereAdi,
    double orneklemeFrekansi) {

    int genislik = 1024;
    int yukseklik = 500;

    cv::Mat grafik(yukseklik, genislik, CV_8UC3,
        cv::Scalar(255, 255, 255));

    double gosterilenFrekans = 100.0;

    int N = (harmonikler.size() - 1) * 2;

    int sonIndeks = std::min(
        (int)(gosterilenFrekans * N / orneklemeFrekansi),
        (int)harmonikler.size() - 1
    );

    double maksimum = 0.0;

    for (int i = 0; i <= sonIndeks; i++) {
        maksimum = std::max(maksimum, harmonikler[i]);
    }

    if (maksimum == 0) maksimum = 1;

    cv::line(grafik,
        cv::Point(0, 450),
        cv::Point(genislik - 1, 450),
        cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i <= sonIndeks; i++) {

        int x = (int)((double)i / sonIndeks * (genislik - 1));

        int yukseklik = (int)(harmonikler[i] / maksimum * 400);

        cv::line(grafik,
            cv::Point(x, 450),
            cv::Point(x, 450 - yukseklik),
            cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow(pencereAdi, grafik);
}

int main() {

    srand((unsigned int)time(0));

    int N = 1024;

    double orneklemeFrekansi = 1024.0;

    double frekans1 = 50.0;
    double genlik1 = 12.0;

    double frekans2 = 30.0;
    double genlik2 = 5.0;

    double gurultuGenligi = 1.0;

    std::vector<double> sinyal1(N);
    std::vector<double> sinyal2(N);

    sinyalOlustur(sinyal1, frekans1, genlik1, orneklemeFrekansi);

    sinyalOlustur(sinyal2, frekans2, genlik2, orneklemeFrekansi);
    gurultuEkle(sinyal1, gurultuGenligi);

    gurultuEkle(sinyal2, gurultuGenligi);

    std::vector<double> harmonik1 = dftAnaliz(sinyal1);

    std::vector<double> harmonik2 = dftAnaliz(sinyal2);

    std::vector<double> toplamSinyal(N);

    for (int i = 0; i < N; i++) {
        toplamSinyal[i] = sinyal1[i] + sinyal2[i];
    }

    std::vector<double> toplamHarmonik = dftAnaliz(toplamSinyal);

    sinyalGrafikCiz(sinyal1, "1 - 50 Hz Sinyal");

    sinyalGrafikCiz(sinyal2, "2 - 30 Hz Sinyal");

    sinyalGrafikCiz(toplamSinyal, "3 - Toplam Sinyal");

    dftGrafikCiz(harmonik1, "4 - Birinci Sinyal DFT", orneklemeFrekansi);

    dftGrafikCiz(harmonik2, "5 - Ikinci Sinyal DFT", orneklemeFrekansi);

    dftGrafikCiz(toplamHarmonik, "6 - Toplam Sinyal DFT", orneklemeFrekansi);

    std::cout << "\nBIRINCI SINYAL DFT SONUCU\n";
    std::cout << "50 Hz harmonik genligi: " << harmonik1[50] << std::endl;

    std::cout << "\nIKINCI SINYAL DFT SONUCU\n";
    std::cout << "30 Hz harmonik genligi: " << harmonik2[30] << std::endl;

    std::cout << "\nTOPLAM SINYAL DFT SONUCU\n";
    std::cout << "30 Hz harmonik genligi: " << toplamHarmonik[30] << std::endl;
    std::cout << "50 Hz harmonik genligi: " << toplamHarmonik[50] << std::endl;

    cv::waitKey(0);

    return 0;
}

