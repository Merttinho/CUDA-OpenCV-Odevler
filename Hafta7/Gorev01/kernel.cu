#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>
#include <opencv2/opencv.hpp>

void sinDalgasi(std::vector<double>& sinyal, double faz, double periyot, double genlik) {

    for (int i = 0; i < sinyal.size(); i++) {

        sinyal[i] = genlik * std::sin((2 * CV_PI * i / periyot) + faz);

    }
}

int main() {

    std::vector<double> sinyal(1024);

    double faz = 0.0;
    double periyot = 128.0;
    double genlik = 100.0;

    sinDalgasi(sinyal, faz, periyot, genlik);

    cv::Mat grafik(400, 1024, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::line(grafik, cv::Point(0, 200), cv::Point(1023, 200),
        cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i < sinyal.size() - 1; i++) {

        int y1 = 200 - static_cast<int>(sinyal[i]);
        int y2 = 200 - static_cast<int>(sinyal[i + 1]);

        cv::line(grafik,
            cv::Point(i, y1),
            cv::Point(i + 1, y2),
            cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Sinus Dalgasi", grafik);
    cv::waitKey(0);

    return 0;
}