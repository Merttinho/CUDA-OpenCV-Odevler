
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

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

    Mat result = Mat::zeros(blurred.size(), CV_8UC1);

    int kernel[3][3] =
    {
        {-1, -1, -1},
        { 0,  0,  0},
        { 1,  1,  1}
    };

    for (int y = 1; y < blurred.rows - 1; y++)
    {
        for (int x = 1; x < blurred.cols - 1; x++)
        {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int pixel = blurred.at<uchar>(y + ky, x + kx);
                    sum += pixel * kernel[ky + 1][kx + 1];
                }
            }

            result.at<uchar>(y, x) =
                static_cast<uchar>(min(abs(sum), 255));
        }
    }

    imwrite("prewitt_sonuc.png", result);

    cout << "Gauss yumusatma ve Prewitt filtresi tamamlandi." << endl;
    cout << "Sonuc: prewitt_sonuc.png" << endl;


    imshow("Orijinal Goruntu", image);
    imshow("Gauss ile Yumusatilmis", blurred);
    imshow("Prewitt Sonucu", result);

    waitKey(0);
    destroyAllWindows();

    return 0;
}