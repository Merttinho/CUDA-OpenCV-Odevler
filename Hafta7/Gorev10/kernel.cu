#include <iostream>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat ikiBoyutluDFT(const Mat& image) {
    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    Mat complexImage;
    dft(floatImage, complexImage, DFT_COMPLEX_OUTPUT);

    return complexImage;
}

void dftShift(Mat& image) {
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    Mat q0(image, Rect(0, 0, cx, cy));
    Mat q1(image, Rect(cx, 0, cx, cy));
    Mat q2(image, Rect(0, cy, cx, cy));
    Mat q3(image, Rect(cx, cy, cx, cy));

    Mat temp;

    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);
}

Mat frekansFiltresi(const Mat& image, int filtreTipi, float D0, float D1) {
    Mat input = image.clone();

    int yeniGenislik = input.cols & -2;
    int yeniYukseklik = input.rows & -2;

    input = input(Rect(0, 0, yeniGenislik, yeniYukseklik));

    Mat complexImage = ikiBoyutluDFT(input);

    dftShift(complexImage);

    vector<Mat> planes;
    split(complexImage, planes);

    Mat mask(input.size(), CV_32F);

    int merkezX = input.cols / 2;
    int merkezY = input.rows / 2;

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            float dx = x - merkezX;
            float dy = y - merkezY;
            float distance = sqrt(dx * dx + dy * dy);

            float deger = 0;

            if (filtreTipi == 0) {
                deger = distance <= D0 ? 1.0f : 0.0f;
            }
            else if (filtreTipi == 1) {
                deger = distance >= D0 ? 1.0f : 0.0f;
            }
            else if (filtreTipi == 2) {
                deger = distance >= D0 && distance <= D1 ? 1.0f : 0.0f;
            }
            else if (filtreTipi == 3) {
                deger = distance < D0 || distance > D1 ? 1.0f : 0.0f;
            }

            mask.at<float>(y, x) = deger;
        }
    }

    multiply(planes[0], mask, planes[0]);
    multiply(planes[1], mask, planes[1]);

    merge(planes, complexImage);

    dftShift(complexImage);

    Mat sonuc;
    idft(complexImage, sonuc, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

    Mat goruntu;
    normalize(sonuc, goruntu, 0, 255, NORM_MINMAX);
    goruntu.convertTo(goruntu, CV_8U);

    return goruntu;
}

void gaussGurultuEkle(const Mat& image, Mat& result, double stddev) {
    Mat noise(image.size(), CV_32F);
    randn(noise, 0, stddev);

    Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    Mat noisyImage = floatImage + noise;
    noisyImage.convertTo(result, CV_8U);
}

void saltPepperGurultuEkle(const Mat& image, Mat& result, double ratio) {
    result = image.clone();

    int toplamPiksel = image.rows * image.cols;
    int gurultuluPiksel = static_cast<int>(toplamPiksel * ratio);

    for (int i = 0; i < gurultuluPiksel; i++) {
        int x = rand() % image.cols;
        int y = rand() % image.rows;

        result.at<uchar>(y, x) = (rand() % 2 == 0) ? 0 : 255;
    }
}

void filtreleriUygula(const Mat& image, const string& isim) {
    Mat lowPass = frekansFiltresi(image, 0, 40, 80);
    Mat highPass = frekansFiltresi(image, 1, 40, 80);
    Mat bandPass = frekansFiltresi(image, 2, 40, 80);
    Mat bandReject = frekansFiltresi(image, 3, 40, 80);

    imshow(isim + " - Low Pass", lowPass);
    imshow(isim + " - High Pass", highPass);
    imshow(isim + " - Band Pass", bandPass);
    imshow(isim + " - Band Reject", bandReject);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    Mat image = imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta7/Gorev10/resim.jpeg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "Goruntu yuklenemedi!" << endl;
        return -1;
    }

    Mat gaussNoise;
    Mat saltPepperNoise;

    gaussGurultuEkle(image, gaussNoise, 25);
    saltPepperGurultuEkle(image, saltPepperNoise, 0.05);

    imshow("Orijinal Goruntu", image);
    imshow("Gaussian Gurultulu", gaussNoise);
    imshow("Salt Pepper Gurultulu", saltPepperNoise);

    filtreleriUygula(image, "Orijinal");
    filtreleriUygula(gaussNoise, "Gaussian");
    filtreleriUygula(saltPepperNoise, "Salt Pepper");

    waitKey(0);
    destroyAllWindows();

    return 0;
}

