#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;


long long dosyaBoyutu(string dosyaAdi) {
    ifstream dosya(dosyaAdi, ios::binary | ios::ate);
    return dosya.tellg();
}

int main() {

    Mat image(1000, 1000, CV_8UC3, Scalar(100, 150, 200));

    imwrite("renkli.bmp", image);

    Mat openedImage = imread("renkli.bmp", IMREAD_COLOR);
    Mat grayImage;

    if (openedImage.empty()) {
        cout << "Goruntu acilamadi!" << endl;
        return -1;
    }

    cv::cvtColor(openedImage,grayImage,COLOR_BGR2GRAY);
    imwrite("Gri.bmp", grayImage);

    cout << "Renkli BMP: " << dosyaBoyutu("renkli.bmp") / 1024.0 << " KB" << endl;
    cout << "Gri BMP: " << dosyaBoyutu("Gri.bmp") / 1024.0 << " KB" << endl;

    imshow("Renkli Goruntu", openedImage);
    imshow("Gri Goruntu", grayImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
