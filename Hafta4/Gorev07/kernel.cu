
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

double elapsedMs(int64 start)
{
    return (getTickCount() - start) * 1000.0 / getTickFrequency();
}

int main()
{
    string imagePath =
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta4/Gorev05/duvar.jpeg";

    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cout << "Resim okunamadi: " << imagePath << endl;
        return -1;
    }

    resize(image, image, Size(1024, 768));

    Mat blurred;

    Mat cannyResult, laplacianResult;
    Mat sobelResult, prewittResult;

    double cannyTime, laplacianTime;
    double sobelTime, prewittTime;

    int64 start = getTickCount();

    GaussianBlur(image, blurred, Size(3, 3), 0);
    Canny(blurred, cannyResult, 50, 150, 3);

    cannyTime = elapsedMs(start);

    start = getTickCount();

    GaussianBlur(image, blurred, Size(3, 3), 0);

    Mat lapKernel = (Mat_<float>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0);

    Mat lapRaw, lapAbs;

    filter2D(blurred, lapRaw, CV_32F, lapKernel);
    convertScaleAbs(lapRaw, lapAbs);
    threshold(lapAbs, laplacianResult, 40, 255, THRESH_BINARY);

    laplacianTime = elapsedMs(start);

    start = getTickCount();

    GaussianBlur(image, blurred, Size(3, 3), 0);

    Mat sobelX, sobelY, sobelMag;

    Sobel(blurred, sobelX, CV_32F, 1, 0, 3);
    Sobel(blurred, sobelY, CV_32F, 0, 1, 3);
    magnitude(sobelX, sobelY, sobelMag);

    threshold(sobelMag, sobelMag, 100, 255, THRESH_BINARY);
    sobelMag.convertTo(sobelResult, CV_8U);

    sobelTime = elapsedMs(start);

    start = getTickCount();

    GaussianBlur(image, blurred, Size(3, 3), 0);

    Mat prewittXKernel = (Mat_<float>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);

    Mat prewittYKernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1);

    Mat prewittX, prewittY, prewittMag;

    filter2D(blurred, prewittX, CV_32F, prewittXKernel);
    filter2D(blurred, prewittY, CV_32F, prewittYKernel);
    magnitude(prewittX, prewittY, prewittMag);

    threshold(prewittMag, prewittMag, 100, 255, THRESH_BINARY);
    prewittMag.convertTo(prewittResult, CV_8U);

    prewittTime = elapsedMs(start);

    imwrite("canny_sonuc.png", cannyResult);
    imwrite("laplacian_sonuc.png", laplacianResult);
    imwrite("sobel_sonuc.png", sobelResult);
    imwrite("prewitt_sonuc.png", prewittResult);

    cout << fixed << setprecision(4);
    cout << "\n===== KENAR TESPITI KARSILASTIRMASI =====\n";

    cout << "Canny sure:      " << cannyTime << " ms\n";
    cout << "Laplacian sure:  " << laplacianTime << " ms\n";
    cout << "Sobel sure:      " << sobelTime << " ms\n";
    cout << "Prewitt sure:    " << prewittTime << " ms\n";

    cout << "\n===== KENAR PIXELI SAYILARI =====\n";
    cout << "Canny:      " << countNonZero(cannyResult) << "\n";
    cout << "Laplacian:  " << countNonZero(laplacianResult) << "\n";
    cout << "Sobel:      " << countNonZero(sobelResult) << "\n";
    cout << "Prewitt:    " << countNonZero(prewittResult) << "\n";

    imshow("Orijinal Goruntu", image);
    imshow("Canny", cannyResult);
    imshow("Laplacian", laplacianResult);
    imshow("Sobel", sobelResult);
    imshow("Prewitt", prewittResult);

    waitKey(0);
    destroyAllWindows();

    return 0;
}