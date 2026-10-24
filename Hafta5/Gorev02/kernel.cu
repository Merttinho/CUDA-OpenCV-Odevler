#include <iostream>
#include <opencv2/opencv.hpp>

void calcHist(cv::Mat image,int hist[256]) {
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int pixel = image.at<uchar>(y, x);
			hist[pixel]++;
		}
	}
}

int main() {
	cv::Mat image = cv::imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta5/Gorev02/goruntu.jpg");
	if (image.empty()) {
		std::cout << "Goruntu okunamadi";
		return -1;
	}

	std::vector<cv::Mat> channels{};

	cv::split(image, channels);

	int histB[256];
	int histG[256];
	int histR[256];

	calcHist(channels[0], histB);
	calcHist(channels[1], histG);
	calcHist(channels[2], histR);

	for (int i = 0; i < 256; i++) {
		std::cout << i << " Piksel yogunlugu" << std::endl;
		std::cout << histB[i] << "Bu yogunluktaki mavi piksel sayisi" << std::endl;
		std::cout << histG[i] << "Bu yogunluktaki yesil piksel sayisi" << std::endl;
		std::cout << histR[i] << " Bu yogunluktaki kirmizi piksel sayisi" << std::endl;
	}
	int histToplamB = 0;
	int histToplamG = 0;
	int histToplamR = 0;
	for (int i = 0; i < 256; i++) {
		histToplamB += histB[i];
		histToplamG += histG[i];
		histToplamR += histR[i];
	}
	std::cout << histToplamB << "Toplam Mavi piksel sayisi" << std::endl;
	std::cout << histToplamG << "Toplam Yesil piksel sayisi" << std::endl;
	std::cout << histToplamR << "Toplam Kirmizi piksel sayisi" << std::endl;

	cv::imshow("fotograf", image);
	cv::waitKey(0);
}




