#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
	cv::Mat image = cv::imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta5/Gorev01/goruntu.jpg");
	cv::Mat resizedImage;

	if (image.empty()) {
		std::cout <<"goruntu okunamadi";
		return -1;
	}
	std::vector<cv::Mat> channels; {}
	cv::split(image, channels);

	int histB[256]{};
	for (int y = 0; y < channels[0].rows;  y++) {
		for (int x = 0; x < channels[0].cols; x++) {
			int pixel = channels[0].at<uchar>(y,x);
			histB[pixel]++;
		}
	}

	int histG[256]{};
	for (int y = 0; y < channels[1].rows; y++) {
		for (int x = 0; x < channels[1].cols; x++) {
			int pixel = channels[1].at<uchar>(y, x);
			histG[pixel]++;
		}
	}

	int histR[256]{};
	for (int y = 0; y < channels[2].rows; y++) {
		for (int x = 0; x < channels[2].cols; x++) {
			int pixel = channels[2].at<uchar>(y, x);
			histR[pixel]++;
		}
		}

	for (int i = 0; i < 256; i++) {
			std::cout << i << " Piksel yogunlugu" << std::endl;
			std::cout << histB[i] <<"Bu yogunluktaki mavi piksel sayisi" << std::endl;
			std::cout << histG[i] << "Bu yogunluktaki yesil piksel sayisi" << std::endl;
			std::cout << histR[i] << " Bu yogunluktaki kirmizi piksel sayisi" << std::endl;
	}





	cv::imshow("Orijinal", image);
	cv::waitKey(0);
}




