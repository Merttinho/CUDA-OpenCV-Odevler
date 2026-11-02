
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

void sinDalgasi(std::vector<double>& sinyal, double faz, double periyot, double genlik) {
    for (int i = 0; i < sinyal.size(); i++) {
        sinyal[i] = genlik * std::sin((2 * CV_PI * i / periyot) + faz);
    }
}

void gurultuEkle(std::vector<double>& sinyal, double gurultuOrani, double genlik) {

    double gurultuGenligi = genlik * gurultuOrani;

    for (int i = 0; i < sinyal.size(); i++) {

        double rastgele = (double)rand() / RAND_MAX;

        double gurultu = (rastgele * 2.0 - 1.0) * gurultuGenligi;

        sinyal[i] = sinyal[i] + gurultu;
    }
}

void grafikCiz(const std::vector<double>& sinyal, std::string pencereAdi) {

    cv::Mat grafik(500, 1024, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::line(grafik, cv::Point(0, 250),
        cv::Point(1023, 250),
        cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i < sinyal.size() - 1; i++) {

        int y1 = 250 - static_cast<int>(sinyal[i]);
        int y2 = 250 - static_cast<int>(sinyal[i + 1]);

        cv::line(grafik,
            cv::Point(i, y1),
            cv::Point(i + 1, y2),
            cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow(pencereAdi, grafik);
}

int main() {

    srand(static_cast<unsigned int>(time(0)));

    std::vector<double> sinyal(1024);

    double faz = 0.0;
    double periyot = 128.0;
    double genlik = 100.0;

    double gurultuOrani = 0.80;

    sinDalgasi(sinyal, faz, periyot, genlik);

    std::vector<double> orijinalSinyal = sinyal;

    std::vector<double> gurultuluSinyal = orijinalSinyal;

    gurultuEkle(gurultuluSinyal, gurultuOrani, genlik);

    std::cout << "ORIJINAL VE GURULTULU SINYAL KARSILASTIRMASI\n\n";

    for (int i = 0; i < 10; i++) {
        std::cout << "Orijinal: " << orijinalSinyal[i]
            << " | Gurultulu: " << gurultuluSinyal[i]
            << std::endl;
    }

    grafikCiz(orijinalSinyal, "Orijinal Sinyal");

    grafikCiz(gurultuluSinyal, "Gurultulu Sinyal");

    cv::waitKey(0);

    return 0;
}
