#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

void gaussianBlur(const cv::Mat& image, cv::Mat& result)
{
	int kernel[5][5] = {
		{ 1,  4,  6,  4, 1 },
		{ 4, 16, 24, 16, 4 },
		{ 6, 24, 36, 24, 6 },
		{ 4, 16, 24, 16, 4 },
		{ 1,  4,  6,  4, 1 }
	};

	result = cv::Mat::zeros(image.size(), CV_8U);
	cv::Mat padded;
	cv::copyMakeBorder(image, padded, 2, 2, 2, 2, cv::BORDER_REFLECT);

	for (int y = 0; y < image.rows; y++) {
		for (int x=0; x < image.cols;x++)
		{
			int sum = 0;
			int weightSum = 0;
			for (int ky = -2; ky <= 2; ky++) {
				for (int kx = -2; kx <= 2; kx++) {
					int pixel = padded.at<uchar>(y + ky+2, x + kx+2);
					int weight = kernel[ky + 2][kx + 2];
					sum += pixel * weight;
					weightSum += weight;
				}
			}
			result.at<uchar>(y, x) = sum / weightSum;
		}
	}
}

void sobelGradients(const cv::Mat& src, cv::Mat& gradX, cv::Mat& gradY) {
	int kernelx[3][3] = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	}; 
	int kernely[3][3] = {
		{-1,-2,-1},
		{0,0,0},
		{1,2,1}
	};
	cv::Mat padded;
	cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_REFLECT);
	gradX = cv::Mat::zeros(src.size(), CV_32F);
	gradY = cv::Mat::zeros(src.size(), CV_32F);
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			float sumX = 0;
			float sumY = 0;
			for (int ky = -1; ky <= 1; ky++) {
				for (int kx = -1; kx <= 1; kx++) {
					int pixel = padded.at<uchar>(y + ky + 1, x + kx + 1);
					sumX += pixel * kernelx[ky + 1][kx + 1];
					sumY += pixel * kernely[ky + 1][kx + 1];
				}
			}
			gradX.at<float>(y, x) = sumX;
			gradY.at<float>(y, x) = sumY;
		}
	}
}

void nonMaximumSuppression(
	const cv::Mat& magnitude,
	const cv::Mat& angle,
	cv::Mat& suppressed)
{
	suppressed = cv::Mat::zeros(magnitude.size(), CV_32F);
	for (int y = 1; y < magnitude.rows - 1; y++)
	{
		for (int x = 1; x < magnitude.cols - 1; x++) {
			float angleValue = angle.at<float>(y, x);
			float current = magnitude.at<float>(y, x);
			if (angleValue < 0) {
				angleValue += 180;
			}
			float neighbor1 = 0;
			float neighbor2 = 0;
			if (angleValue < 22.5 || angleValue >= 157.5)
			{
				neighbor1 = magnitude.at<float>(y, x - 1);
				neighbor2 = magnitude.at<float>(y, x + 1);
			}
			else if (angleValue < 67.5)
			{
				neighbor1 = magnitude.at<float>(y - 1, x - 1);
				neighbor2 = magnitude.at<float>(y + 1, x + 1);
			}
			else if (angleValue < 112.5)
			{
				neighbor1 = magnitude.at<float>(y - 1, x);
				neighbor2 = magnitude.at<float>(y + 1, x);
			}
			else
			{
				neighbor1 = magnitude.at<float>(y - 1, x + 1);
				neighbor2 = magnitude.at<float>(y + 1, x - 1);
			}
			if (current >= neighbor1 && current >= neighbor2)
			{
				suppressed.at<float>(y, x) = current;
			}
			else
			{
				suppressed.at<float>(y, x) = 0;
			}
		}
	}
}

void doubleThreshold(
	const cv::Mat& suppressed,
	cv::Mat& thresholded,
	float lowThreshold,
	float highThreshold)
{
	thresholded = cv::Mat::zeros(suppressed.size(), CV_8U);

	for (int y = 0; y < suppressed.rows; y++)
	{
		for (int x = 0; x < suppressed.cols; x++)
		{
			float pixel = suppressed.at<float>(y, x);

			if (pixel >= highThreshold)
			{
				thresholded.at<uchar>(y, x) = 255;
			}
			else if (pixel >= lowThreshold)
			{
				thresholded.at<uchar>(y, x) = 128;
			}
			else
			{
				thresholded.at<uchar>(y, x) = 0;
			}
		}
	}
}

void hysteresisThreshold(const cv::Mat& thresholded, cv::Mat& edges)
{
	edges = thresholded.clone();

	bool changed = true;

	while (changed)
	{
		changed = false;

		for (int y = 1; y < edges.rows - 1; y++)
		{
			for (int x = 1; x < edges.cols - 1; x++)
			{
				if (edges.at<uchar>(y, x) == 128)
				{
					bool connectedToStrong = false;

					for (int ky = -1; ky <= 1; ky++)
					{
						for (int kx = -1; kx <= 1; kx++)
						{
							if (edges.at<uchar>(y + ky, x + kx) == 255)
							{
								connectedToStrong = true;
							}
						}
					}

					if (connectedToStrong)
					{
						edges.at<uchar>(y, x) = 255;
						changed = true;
					}
				}
			}
		}
	}

	for (int y = 0; y < edges.rows; y++)
	{
		for (int x = 0; x < edges.cols; x++)
		{
			if (edges.at<uchar>(y, x) != 255)
			{
				edges.at<uchar>(y, x) = 0;
			}
		}
	}
}

int main()
{
	cv::Mat image;
	image = cv::imread("C:/Users/USER/Desktop/GoruntuIsleme/Hafta4/OdevCanny/duvar.jpeg", cv::IMREAD_GRAYSCALE); 
	
	if (image.empty()) {
		std::cout << "Goruntu Bos" << std::endl;
		return -1;
	}
	cv::Mat blurred;
	gaussianBlur(image, blurred);

	cv::Mat gradX, gradY;
	sobelGradients(blurred, gradX, gradY);

	cv::Mat magnitude = cv::Mat::zeros(image.size(), CV_32F);
	cv::Mat angle = cv::Mat::zeros(image.size(), CV_32F);

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			float gx = gradX.at<float>(y, x);
			float gy = gradY.at<float>(y, x);
			 magnitude.at<float>(y,x) = std::sqrt(gx * gx + gy * gy);
			 angle.at<float>(y,x) = std::atan2(gy, gx) * 180 / CV_PI;
		}
	}
	cv::Mat magnitudeDisplay;
	cv::normalize(magnitude, magnitudeDisplay, 0, 255, cv::NORM_MINMAX);
	magnitudeDisplay.convertTo(magnitudeDisplay, CV_8U);
	
	cv::Mat suppressed;
	nonMaximumSuppression(magnitude, angle, suppressed);
	cv::Mat suppressedDisplay;
	cv::normalize(suppressed, suppressedDisplay, 0, 255, cv::NORM_MINMAX);
	suppressedDisplay.convertTo(suppressedDisplay, CV_8U);
	cv::Mat thresholded;

	doubleThreshold(suppressed, thresholded, 50, 150);
	cv::Mat edges;
	hysteresisThreshold(thresholded, edges);

	

	
	
	cv::imshow("Orijinal", image);
	cv::imshow("Gradyan Buyuklugu", magnitudeDisplay);
	cv::imshow("NMS Sonucu", suppressedDisplay);
	cv::imshow("Double Threshold", thresholded);
	cv::imshow("Final Canny", edges);

	cv::waitKey(0);

	return 0;
}




