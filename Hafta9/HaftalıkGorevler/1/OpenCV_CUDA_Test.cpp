
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

    VideoCapture camera(0);

    if (!camera.isOpened()) {
        cout << "Kamera acilamadi!" << endl;
        return -1;
    }

    Mat frame, gray, previousGray, difference, thresholdImage;
    Mat ycrcb, skinMask, result;

    while (true) {

        camera >> frame;

        if (frame.empty()) {
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);

        if (previousGray.empty()) {
            gray.copyTo(previousGray);
            imshow("Webcam", frame);

            if (waitKey(1) == 27)
                break;

            continue;
        }

        absdiff(previousGray, gray, difference);

        threshold(difference, thresholdImage, 25, 255, THRESH_BINARY);

        dilate(thresholdImage, thresholdImage, Mat(), Point(-1, -1), 3);

        vector<vector<Point>> contours;

        findContours(thresholdImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        frame.copyTo(result);

        int minArea = 1500;

        for (int i = 0; i < contours.size(); i++) {

            double area = contourArea(contours[i]);

            if (area > minArea) {

                Rect roi = boundingRect(contours[i]);

                roi &= Rect(0, 0, frame.cols, frame.rows);

                rectangle(result, roi, Scalar(255, 0, 0), 2);

                Mat roiImage = frame(roi);

                Mat roiYCrCb;
                Mat roiSkinMask;

                cvtColor(roiImage, roiYCrCb, COLOR_BGR2YCrCb);

                inRange(roiYCrCb, Scalar(0, 133, 77), Scalar(255, 173, 127), roiSkinMask);

                Mat coloredSkin = Mat::zeros(roiImage.size(), roiImage.type());

                coloredSkin.setTo(Scalar(0, 255, 0), roiSkinMask);

                coloredSkin.copyTo(result(roi), roiSkinMask);
            }
        }

        gray.copyTo(previousGray);

        imshow("Webcam", frame);
        imshow("Hareket Maskesi", thresholdImage);
        imshow("Hareket ve ROI - Ten Tespiti", result);

        if (waitKey(1) == 27) {
            break;
        }
    }

    camera.release();
    destroyAllWindows();

    return 0;
}
