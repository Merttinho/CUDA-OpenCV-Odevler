
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

    Mat image = imread("insan.jpeg", IMREAD_COLOR);

    if (image.empty()) {
        cout << "Goruntu acilamadi!" << endl;
        return -1;
    }

    Mat ycrcbImage;
    Mat hsvImage;

    cvtColor(image, ycrcbImage, COLOR_BGR2YCrCb);
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    Mat rgbResult = image.clone();
    Mat ycrcbResult = image.clone();
    Mat hsvResult = image.clone();

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            Vec3b pixel = image.at<Vec3b>(y, x);

            int B = pixel[0];
            int G = pixel[1];
            int R = pixel[2];

            bool rgbSkin =
                R > 95 &&
                G > 40 &&
                B > 20 &&
                R > G &&
                R > B;

            if (!rgbSkin) {
                rgbResult.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }

            Vec3b ycrcbPixel = ycrcbImage.at<Vec3b>(y, x);

            int Y = ycrcbPixel[0];
            int Cr = ycrcbPixel[1];
            int Cb = ycrcbPixel[2];

            bool ycrcbSkin =
                Cb >= 77 && Cb <= 127 &&
                Cr >= 133 && Cr <= 173;

            if (!ycrcbSkin) {
                ycrcbResult.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }

            Vec3b hsvPixel = hsvImage.at<Vec3b>(y, x);

            int H = hsvPixel[0];
            int S = hsvPixel[1];
            int V = hsvPixel[2];

            bool hsvSkin =
                ((H >= 0 && H <= 25) || (H >= 170 && H <= 180)) &&
                S >= 30 &&
                V >= 50;

            if (!hsvSkin) {
                hsvResult.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }

    imwrite("RGB_Ten_Tespiti.bmp", rgbResult);
    imwrite("YCbCr_Ten_Tespiti.bmp", ycrcbResult);
    imwrite("HSV_Ten_Tespiti.bmp", hsvResult);

    imshow("Orijinal Goruntu", image);
    imshow("RGB Ten Tespiti", rgbResult);
    imshow("YCbCr Ten Tespiti", ycrcbResult);
    imshow("HSV Ten Tespiti", hsvResult);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
