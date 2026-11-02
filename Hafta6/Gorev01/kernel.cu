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

	std::vector<cv::Vec2f> lines;

	cv::HoughLines(medianEdges, lines, 1, CV_PI / 180, 45);
	std::cout << "Bulunan dogru sayisi: " << lines.size() << std::endl;

	cv::Mat houghResult = image.clone();
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];

		double a = cos(theta);
		double b = sin(theta);

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
		cv::line(houghResult, pt1, pt2, cv::Scalar(0, 0, 255), 2);
	}

	std::vector<cv::Vec4i> linesP;
	cv::HoughLinesP(medianEdges, linesP, 1, CV_PI / 180, 44, 200, 70);
	cv::Mat houghPResult = image.clone();
	for (size_t i = 0; i < linesP.size(); i++) {
		cv::Point ptP1(
			linesP[i][0],linesP[i][1]
		);
		cv::Point ptP2(
			linesP[i][2], linesP[i][3]
		);
		cv::line(houghPResult, ptP1, ptP2, cv::Scalar(0, 255, 0), 2);
	}



	cv::imshow("HoughLines", houghResult);
	cv::imshow("HoughPLines", houghPResult);
	cv::waitKey(0);
}