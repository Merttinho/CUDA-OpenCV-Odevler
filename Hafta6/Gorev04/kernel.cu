#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main() {

    cv::Mat image = cv::imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta6/Gorev01/NoıseandLınes.jpeg"
    );

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

    double cosTable[360];
    double sinTable[360];

    for (int t = 0; t < 360; t++) {
        double angle = t * CV_PI / 180.0;

        cosTable[t] = cos(angle);
        sinTable[t] = sin(angle);
    }

    int minRadius = 10;
    int maxRadius = 100;

    int rows = image.rows;
    int cols = image.cols;
    int radiusCount = maxRadius - minRadius + 1;

    std::vector<std::vector<std::vector<int>>> accumulator(
        rows,
        std::vector<std::vector<int>>(
            cols,
            std::vector<int>(radiusCount, 0)
        )
    );

    for (int y = 0; y < medianEdges.rows; y++) {
        for (int x = 0; x < medianEdges.cols; x++) {

            if (medianEdges.at<uchar>(y, x) == 255) {

                for (int r = minRadius; r <= maxRadius; r++) {
                    for (int t = 0; t < 360; t++) {

                        int a = cvRound(x - r * cosTable[t]);
                        int b = cvRound(y - r * sinTable[t]);

                        if (a >= 0 && a < cols &&
                            b >= 0 && b < rows) {

                            accumulator[b][a][r - minRadius]++;
                        }
                    }
                }
            }
        }
    }

    int maxVotes = 0;

    for (int b = 0; b < rows; b++) {
        for (int a = 0; a < cols; a++) {
            for (int r = 0; r < radiusCount; r++) {

                maxVotes = std::max(
                    maxVotes,
                    accumulator[b][a][r]
                );
            }
        }
    }

    int threshold = 100;

    cv::Mat result = image.clone();

    for (int b = 0; b < rows; b++) {
        for (int a = 0; a < cols; a++) {
            for (int r = 0; r < radiusCount; r++) {

                if (accumulator[b][a][r] >= threshold) {

                    int radius = r + minRadius;

                    bool localMaximum = true;

                    for (int db = -3; db <= 3 && localMaximum; db++) {
                        for (int da = -3; da <= 3 && localMaximum; da++) {
                            for (int dr = -2; dr <= 2; dr++) {

                                int nb = b + db;
                                int na = a + da;
                                int nr = r + dr;

                                if (nb < 0 || nb >= rows ||
                                    na < 0 || na >= cols ||
                                    nr < 0 || nr >= radiusCount) {
                                    continue;
                                }

                                if (accumulator[nb][na][nr] >
                                    accumulator[b][a][r]) {
                                    localMaximum = false;
                                    break;
                                }
                            }
                        }
                    }

                    if (localMaximum) {
                        cv::circle(
                            result,
                            cv::Point(a, b),
                            radius,
                            cv::Scalar(0, 0, 255),
                            2
                        );
                    }
                }
            }
        }
    }


    cv::imshow("Original Image", image);
    cv::imshow("Canny Edges", medianEdges);
    cv::imshow("Detected Circles", result);

    cv::waitKey(0);

    return 0;
}
