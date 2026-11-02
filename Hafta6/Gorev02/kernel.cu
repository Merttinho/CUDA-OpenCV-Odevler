#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
	cv::Mat image = cv::imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta6/Gorev01/NoıseandLınes.jpeg");
	if (image.empty()) {
		std::cout << "Goruntu Okunamadi";
		return -1;
	}
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat edges;
	cv::Canny(grayImage, edges, 50, 150);

	cv::Mat medianEdges;
	cv::medianBlur(edges, medianEdges, 3);

    double cosTable[180];
    double sinTable[180];

    for (int i = 0; i < 180; i++) {
        double theta = i * CV_PI / 180.0;

        cosTable[i] = cos(theta);
        sinTable[i] = sin(theta);
    }


    int rhoMax = cvRound(std::sqrt(
        std::pow(image.cols, 2) + std::pow(image.rows, 2)
    ));

    int rhoSize = 2 * rhoMax + 1;

    std::vector<std::vector<int>> accumulator(
        180, std::vector<int>(rhoSize, 0)
    );

    for (int y = 0; y < medianEdges.rows; y++) {
        for (int x = 0; x < medianEdges.cols; x++) {

            if (medianEdges.at<uchar>(y, x) == 255) {

                for (int t = 0; t < 180; t++) {

                    double rho = x * cosTable[t] + y * sinTable[t];

                    int rhoIndex = cvRound(rho) + rhoMax;

                    accumulator[t][rhoIndex]++;
                }
            }
        }
    }


    cv::Mat accumulatorMat(180, rhoSize, CV_32S);

    for (int t = 0; t < 180; t++) {
        for (int r = 0; r < rhoSize; r++) {
            accumulatorMat.at<int>(t, r) = accumulator[t][r];
        }
    }


    cv::Mat accumulatorImage;

    cv::normalize(
        accumulatorMat,
        accumulatorImage,
        0,
        255,
        cv::NORM_MINMAX,
        CV_8U
    );


    cv::imwrite("HoughAccumulator.jpg", accumulatorImage);

    cv::imshow("Accumulator", accumulatorImage);


    int threshold = 45;

    cv::Mat result = image.clone();

    for (int t = 0; t < 180; t++) {
        for (int r = 0; r < rhoSize; r++) {

            if (accumulator[t][r] >= threshold) {

                double rho = r - rhoMax;

                double a = cosTable[t];
                double b = sinTable[t];

                double x0 = a * rho;
                double y0 = b * rho;

                cv::Point pt1(
                    cvRound(x0 + 1000 * (-b)),
                    cvRound(y0 + 1000 * a)
                );

                cv::Point pt2(
                    cvRound(x0 - 1000 * (-b)),
                    cvRound(y0 - 1000 * a)
                );

                cv::line(
                    result,
                    pt1,
                    pt2,
                    cv::Scalar(0, 0, 255),
                    2
                );
            }
        }
    }

    cv::imshow("Manual Hough Transform", result);

    cv::waitKey(0);
}