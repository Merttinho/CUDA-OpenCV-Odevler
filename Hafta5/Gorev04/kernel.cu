
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <array>

using ColorConvertFn = cv::Mat(*)(const cv::Mat&);

cv::Mat myBGRtoHSV(const cv::Mat& image)
{
    constexpr int shift = 12;

    static const std::array<int, 256> sdiv = [] {
        std::array<int, 256> table{};
        for (int i = 1; i < 256; i++)
            table[i] = cv::saturate_cast<int>((255 << 12) / (1.0 * i));
        return table;
        }();

    static const std::array<int, 256> hdiv = [] {
        std::array<int, 256> table{};
        for (int i = 1; i < 256; i++)
            table[i] = cv::saturate_cast<int>((180 << 12) / (6.0 * i));
        return table;
        }();

    cv::Mat hsv(image.rows, image.cols, CV_8UC3);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];

            int v = std::max({ r, g, b });
            int vmin = std::min({ r, g, b });
            int diff = v - vmin;

            int s = (diff * sdiv[v] + (1 << (shift - 1))) >> shift;

            int vr = (v == r) ? -1 : 0;
            int vg = (v == g) ? -1 : 0;

            int h = (vr & (g - b)) +
                (~vr & (
                    (vg & (b - r + 2 * diff)) +
                    (~vg & (r - g + 4 * diff))
                    ));

            h = (h * hdiv[diff] + (1 << (shift - 1))) >> shift;

            if (h < 0)
                h += 180;

            hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(h),
                static_cast<uchar>(s),
                static_cast<uchar>(v)
            );
        }
    }

    return hsv;
}
int main() {
    cv::Mat image = cv::imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta5/Gorev02/goruntu.jpg"
    );

    if (image.empty()) {
        std::cout << "Goruntu okunamadi.\n";
        return -1;
    }

    cv::Mat hsvOpenCV;
    cv::cvtColor(image, hsvOpenCV, cv::COLOR_BGR2HSV);

    ColorConvertFn convert = myBGRtoHSV;
    cv::Mat hsvMy = convert(image);

    long long toplamHata = 0;
    int farkliKanalSayisi = 0;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b a = hsvOpenCV.at<cv::Vec3b>(y, x);
            cv::Vec3b b = hsvMy.at<cv::Vec3b>(y, x);

            for (int c = 0; c < 3; c++) {
                int fark = std::abs(
                    static_cast<int>(a[c]) -
                    static_cast<int>(b[c])
                );

                toplamHata += fark;

                if (fark != 0) {
                    farkliKanalSayisi++;
                }
            }
        }
    }

    std::cout << "Toplam hata: " << toplamHata << '\n';
    std::cout << "Farkli kanal sayisi: " << farkliKanalSayisi << '\n';

    cv::imshow("OpenCV HSV", hsvOpenCV);
    cv::imshow("Manuel HSV", hsvMy);
    cv::waitKey(0);

    return 0;
}

