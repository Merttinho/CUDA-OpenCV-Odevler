
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

int main() {

    Mat image(4320, 7680, CV_8UC3, Scalar(100, 150, 200));

    imwrite("8K_Goruntu.bmp", image);

    Mat openedImage = imread("8K_Goruntu.bmp", IMREAD_COLOR);

    if (openedImage.empty()) {
        cout << "Goruntu acilamadi!" << endl;
        return -1;
    }

    auto baslangic = chrono::high_resolution_clock::now();

    for (int y = 0; y < openedImage.rows; y++) {
        for (int x = 0; x < openedImage.cols; x++) {

            Vec3b pixel = openedImage.at<Vec3b>(y, x);

            pixel[0] /= 2;
            pixel[1] /= 2;
            pixel[2] /= 2;

            openedImage.at<Vec3b>(y, x) = pixel;
        }
    }

    auto bitis = chrono::high_resolution_clock::now();

    double sure = chrono::duration<double, milli>(bitis - baslangic).count();

    imwrite("8K_Duzenlenmis.bmp", openedImage);

    cout << "Goruntu genisligi: " << openedImage.cols << endl;
    cout << "Goruntu yuksekligi: " << openedImage.rows << endl;
    cout << "Kanal sayisi: " << openedImage.channels() << endl;

    cout << "Piksel duzenleme suresi: " << sure << " ms" << endl;

    imshow("8K Duzenlenmis Goruntu", openedImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}

