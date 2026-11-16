#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

int main() {

    Mat image = imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta9/Gorev02/renkli.bmp", IMREAD_COLOR);

    if (image.empty()) {
        cout << "Goruntu acilamadi!" << endl;
        return -1;
    }

    Mat hsvImage;
    Mat yuvImage;
    Mat labImage;
    Mat hsiImage(image.size(), CV_8UC3);

    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    cvtColor(image, yuvImage, COLOR_BGR2YUV);
    cvtColor(image, labImage, COLOR_BGR2Lab);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            Vec3b pixel = image.at<Vec3b>(y, x);

            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double toplam = R + G + B;

            double H = 0;
            double S = 0;
            double I = toplam / 3.0;

            double payda = sqrt((R - G) * (R - G) + (R - B) * (G - B));

            if (payda > 0.000001) {

                double theta = acos(
                    (0.5 * ((R - G) + (R - B))) / payda
                );

                H = theta * 180.0 / CV_PI;

                if (B > G) {
                    H = 360.0 - H;
                }
            }

            if (toplam > 0) {
                S = 1.0 - (3.0 * min(R, min(G, B)) / toplam);
            }

            uchar h = saturate_cast<uchar>(H * 255.0 / 360.0);
            uchar s = saturate_cast<uchar>(S * 255.0);
            uchar i = saturate_cast<uchar>(I);

            hsiImage.at<Vec3b>(y, x) = Vec3b(h, s, i);
        }
    }

    imwrite("HSV.bmp", hsvImage);
    imwrite("YUV.bmp", yuvImage);
    imwrite("Lab.bmp", labImage);
    imwrite("HSI.bmp", hsiImage);

    imshow("Orijinal Goruntu", image);
    imshow("HSV Goruntu", hsvImage);
    imshow("YUV Goruntu", yuvImage);
    imshow("Lab Goruntu", labImage);
    imshow("HSI Goruntu", hsiImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}