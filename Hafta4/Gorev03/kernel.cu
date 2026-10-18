
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat image = imread("duvar.jpeg", IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cout << "Goruntu okunamadi!" << endl;
        return -1;
    }

    resize(image, image, Size(1024, 768));

    Mat blurred;
    GaussianBlur(image, blurred, Size(3, 3), 0.8);

    Mat sobelX, sobelY;
    Mat absX, absY;

    Sobel(blurred, sobelX, CV_16S, 1, 0, 3);
    Sobel(blurred, sobelY, CV_16S, 0, 1, 3);

    convertScaleAbs(sobelX, absX);
    convertScaleAbs(sobelY, absY);

    Mat combined;
    addWeighted(absX, 0.5, absY, 0.5, 0, combined);

    imwrite("sobel_yatay.png", absY);
    imwrite("sobel_dikey.png", absX);
    imwrite("sobel_birlesik.png", combined);

    cout << "Gauss ve Sobel filtreleri uygulandi." << endl;
    cout << "sobel_yatay.png kaydedildi." << endl;
    cout << "sobel_dikey.png kaydedildi." << endl;
    cout << "sobel_birlesik.png kaydedildi." << endl;

    imshow("Orijinal Goruntu", image);
    imshow("Gauss ile Yumusatilmis", blurred);
    imshow("Yatay Sobel", absY);
    imshow("Dikey Sobel", absX);
    imshow("Birlesik Sobel", combined);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
