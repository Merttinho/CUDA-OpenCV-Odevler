
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat edgeMagnitude(Mat gx, Mat gy)
{
    Mat mag, result;
    magnitude(gx, gy, mag);
    normalize(mag, result, 0, 255, NORM_MINMAX, CV_8U);
    return result;
}

Mat zeroCrossing(const Mat& response, float threshold)
{
    Mat edges = Mat::zeros(response.size(), CV_8U);

    for (int y = 1; y < response.rows - 1; y++)
    {
        for (int x = 1; x < response.cols - 1; x++)
        {
            float minVal = response.at<float>(y, x);
            float maxVal = minVal;

            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    float v = response.at<float>(y + dy, x + dx);

                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
            }

            if (minVal < 0 && maxVal > 0 &&
                (maxVal - minVal) >= threshold)
            {
                edges.at<uchar>(y, x) = 255;
            }
        }
    }

    return edges;
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
    GaussianBlur(image, blurred, Size(3, 3), 0);

    Mat logKernel = (Mat_<float>(7, 7) <<
        0, 0, -1, -1, -1, 0, 0,
        0, -2, -3, -3, -3, -2, 0,
        -1, -3, 5, 5, 5, -3, -1,
        -1, -3, 5, 16, 5, -3, -1,
        -1, -3, 5, 5, 5, -3, -1,
        0, -2, -3, -3, -3, -2, 0,
        0, 0, -1, -1, -1, 0, 0);

    Mat logResponse, logAbs, logEdges;

    filter2D(image, logResponse, CV_32F, logKernel);
    convertScaleAbs(logResponse, logAbs);
    logEdges = zeroCrossing(logResponse, 30.0f);

    Mat lapKernel = (Mat_<float>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0);

    Mat lapRaw, lapBlur, lapRawResult, lapBlurResult;

    filter2D(image, lapRaw, CV_32F, lapKernel);
    filter2D(blurred, lapBlur, CV_32F, lapKernel);

    convertScaleAbs(lapRaw, lapRawResult);
    convertScaleAbs(lapBlur, lapBlurResult);

    Mat sxRaw, syRaw, sobelRaw;
    Sobel(image, sxRaw, CV_32F, 1, 0, 3);
    Sobel(image, syRaw, CV_32F, 0, 1, 3);
    sobelRaw = edgeMagnitude(sxRaw, syRaw);

    Mat sxBlur, syBlur, sobelBlur;
    Sobel(blurred, sxBlur, CV_32F, 1, 0, 3);
    Sobel(blurred, syBlur, CV_32F, 0, 1, 3);
    sobelBlur = edgeMagnitude(sxBlur, syBlur);

    Mat pxKernel = (Mat_<float>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);

    Mat pyKernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1);

    Mat pxRaw, pyRaw;
    filter2D(image, pxRaw, CV_32F, pxKernel);
    filter2D(image, pyRaw, CV_32F, pyKernel);
    Mat prewittRaw = edgeMagnitude(pxRaw, pyRaw);

    Mat pxBlur, pyBlur;
    filter2D(blurred, pxBlur, CV_32F, pxKernel);
    filter2D(blurred, pyBlur, CV_32F, pyKernel);
    Mat prewittBlur = edgeMagnitude(pxBlur, pyBlur);

    imwrite("01_gauss.png", blurred);
    imwrite("02_log_abs.png", logAbs);
    imwrite("03_log_zero_crossing.png", logEdges);
    imwrite("04_laplacian_gauss_yok.png", lapRawResult);
    imwrite("05_laplacian_gauss_var.png", lapBlurResult);
    imwrite("06_sobel_gauss_yok.png", sobelRaw);
    imwrite("07_sobel_gauss_var.png", sobelBlur);
    imwrite("08_prewitt_gauss_yok.png", prewittRaw);
    imwrite("09_prewitt_gauss_var.png", prewittBlur);

    imshow("Orijinal", image);
    imshow("Gauss 3x3", blurred);
    imshow("LoG 7x7 - Mutlak yanit", logAbs);
    imshow("LoG 7x7 - Zero crossing", logEdges);
    imshow("Laplacian - Gauss yok", lapRawResult);
    imshow("Laplacian - Gauss var", lapBlurResult);
    imshow("Sobel - Gauss yok", sobelRaw);
    imshow("Sobel - Gauss var", sobelBlur);
    imshow("Prewitt - Gauss yok", prewittRaw);
    imshow("Prewitt - Gauss var", prewittBlur);

    cout << "LoG ve diger kenar filtreleri tamamlandi." << endl;
    cout << "Sonuclar programin calisma klasorune kaydedildi." << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}
