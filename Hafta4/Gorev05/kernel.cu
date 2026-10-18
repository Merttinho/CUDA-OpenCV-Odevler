#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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
    GaussianBlur(image, blurred, Size(3, 3), 0);

    Mat laplacianKernel = (Mat_<float>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0);

    Mat laplacianRaw, laplacianResult;

    filter2D(blurred, laplacianRaw, CV_32F, laplacianKernel);
    convertScaleAbs(laplacianRaw, laplacianResult);

    Mat sobelX, sobelY, sobelMagnitude, sobelResult;

    Sobel(blurred, sobelX, CV_32F, 1, 0, 3);
    Sobel(blurred, sobelY, CV_32F, 0, 1, 3);

    magnitude(sobelX, sobelY, sobelMagnitude);

    normalize(sobelMagnitude, sobelResult,
        0, 255, NORM_MINMAX, CV_8U);

    Mat prewittXKernel = (Mat_<float>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);

    Mat prewittYKernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1);

    Mat prewittX, prewittY;
    Mat prewittMagnitude, prewittResult;

    filter2D(blurred, prewittX, CV_32F, prewittXKernel);
    filter2D(blurred, prewittY, CV_32F, prewittYKernel);

    magnitude(prewittX, prewittY, prewittMagnitude);

    normalize(prewittMagnitude, prewittResult,
        0, 255, NORM_MINMAX, CV_8U);

    imwrite("gauss_yumusatilmis.png", blurred);
    imwrite("laplacian_sonuc.png", laplacianResult);
    imwrite("sobel_sonuc.png", sobelResult);
    imwrite("prewitt_sonuc.png", prewittResult);

    imshow("Orijinal Goruntu", image);
    imshow("Gauss Yumusatma", blurred);
    imshow("Laplacian 3x3", laplacianResult);
    imshow("Sobel", sobelResult);
    imshow("Prewitt", prewittResult);

    cout << "Filtreleme islemleri tamamlandi." << endl;
    cout << "Sonuclar programin calisma klasorune kaydedildi."
        << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}
