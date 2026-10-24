#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat myGray(cv::Mat image) {
	cv::Mat gray(image.rows, image.cols, CV_8UC1);

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

			uchar grayValue = static_cast<uchar>(
				pixel[0] * 0.114f +
				pixel[1] * 0.587f +
				pixel[2] * 0.299f
				);

			gray.at<uchar>(y, x) = grayValue;

		
		}
	}
	return gray;
}

int main() {
	cv::Mat image = cv::imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta5/Gorev02/goruntu.jpg");
	cv::Mat grayImage1;
	cv::Mat grayImage2 = myGray(image);
	if (image.empty()) {
		std::cout << "Goruntu Okunamadi";
		return-1;
	}
	cv::cvtColor(image, grayImage1, cv::COLOR_BGR2GRAY);


	long long hata = 0;
	int farkliPikselSayisi = 0;
	for (int y = 0; y < grayImage1.rows; y++) {
		for (int x = 0; x < grayImage1.cols; x++) {
			hata += std::abs(grayImage1.at<uchar>(y, x) - grayImage2.at<uchar>(y, x));
			if (grayImage1.at<uchar>(y, x) != grayImage2.at<uchar>(y, x)) {
				farkliPikselSayisi++;
			}
		}
	}
	std::cout << "Toplam mutlak piksel hatasi: " << hata << std::endl;
	std::cout << "Farkli piksel sayisi: " << farkliPikselSayisi << std::endl;

	cv::imshow("HazırFonk", grayImage1);
	cv::imshow("BizimFonk", grayImage2);
	cv::waitKey(0);
}




