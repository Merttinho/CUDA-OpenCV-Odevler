
#include <iostream>
#include <opencv2/opencv.hpp>
#include <direct.h>

using namespace cv;
using namespace std;

int main() {

    string root = "C:/Users/USER/Desktop/GoruntuIsleme/Hafta2/Gorev02/OpenCV_CUDA_Test/archive/dataset";
    string outputRoot = root + "/sonuclar";

    _mkdir(outputRoot.c_str());

    vector<String> classes = {
        "speed_limit_30",
        "turn_left",
        "turn_right"
    };

    int toplamGoruntu = 0;
    int tespitEdilen = 0;

    for (int i = 0; i < classes.size(); i++) {

        string folder = root + "/" + classes[i] + "/Germany";
        string outputFolder = outputRoot + "/" + classes[i];

        _mkdir(outputFolder.c_str());

        vector<String> files;
        vector<String> jpegFiles;

        glob(folder + "/*.jpg", files, false);
        glob(folder + "/*.jpeg", jpegFiles, false);

        files.insert(files.end(), jpegFiles.begin(), jpegFiles.end());

        cout << "\nSinif: " << classes[i] << endl;
        cout << "Bulunan goruntu: " << files.size() << endl;

        for (int j = 0; j < files.size(); j++) {

            Mat image = imread(files[j]);

            if (image.empty()) {
                continue;
            }

            Mat hsvImage;
            Mat redMask1, redMask2, redMask;
            Mat result;

            cvtColor(image, hsvImage, COLOR_BGR2HSV);

            inRange(hsvImage, Scalar(0, 80, 50), Scalar(10, 255, 255), redMask1);
            inRange(hsvImage, Scalar(170, 80, 50), Scalar(179, 255, 255), redMask2);

            bitwise_or(redMask1, redMask2, redMask);

            GaussianBlur(redMask, redMask, Size(9, 9), 2);

            vector<Vec3f> circles;

            HoughCircles(
                redMask,
                circles,
                HOUGH_GRADIENT,
                1,
                30,
                100,
                15,
                10,
                150
            );

            result = image.clone();

            for (int k = 0; k < circles.size(); k++) {

                Point center(
                    cvRound(circles[k][0]),
                    cvRound(circles[k][1])
                );

                int radius = cvRound(circles[k][2]);

                circle(result, center, radius, Scalar(0, 255, 0), 3);
                circle(result, center, 3, Scalar(255, 0, 0), -1);
            }

            size_t fileStart = files[j].find_last_of("/\\");
            string fileName = files[j].substr(fileStart + 1);

            string outputPath = outputFolder + "/" + fileName;

            imwrite(outputPath, result);

            toplamGoruntu++;

            if (!circles.empty()) {
                tespitEdilen++;
            }

            cout << fileName
                << " | Daire sayisi: " << circles.size()
                << endl;
        }
    }

    cout << "\n========== SONUC ==========" << endl;
    cout << "Toplam islenen goruntu: " << toplamGoruntu << endl;
    cout << "Daire tespit edilen goruntu: " << tespitEdilen << endl;

    return 0;
}
