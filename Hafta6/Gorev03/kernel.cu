#include <opencv2/opencv.hpp>
#include <iostream>

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
	cv::Mat medianImage;
	cv::medianBlur(grayImage, medianImage, 3);

	std::vector<cv::Vec3f> circles;
	
	cv::HoughCircles(
		medianImage,
		circles,
		cv::HOUGH_GRADIENT,
		1,
		50,
		100,
		30,
		0,
		0
	);
	cv::Mat result = image.clone();
	for (int i = 0; i < circles.size(); i++) {
		cv::Vec3f c = circles[i];
		cv::Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		cv::circle(result, center, radius, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("Hough Circles", result);
	cv::waitKey(0);
}