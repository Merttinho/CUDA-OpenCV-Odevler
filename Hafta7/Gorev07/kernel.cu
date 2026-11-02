#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat ikiBoyutluDFT(const cv::Mat& image)
{
    cv::Mat imageFloat;

    image.convertTo(imageFloat, CV_32F);

    cv::Mat complexImage;

    cv::dft(
        imageFloat,
        complexImage,
        cv::DFT_COMPLEX_OUTPUT
    );

    return complexImage;
}

cv::Mat harmonikGoster(
    const cv::Mat& complexImage)
{
    cv::Mat bilesenler[2];

    cv::split(
        complexImage,
        bilesenler
    );

    cv::Mat magnitude;

    cv::magnitude(
        bilesenler[0],
        bilesenler[1],
        magnitude
    );

    magnitude += 1;

    cv::log(
        magnitude,
        magnitude
    );

    magnitude(
        cv::Rect(
            0,
            0,
            magnitude.cols & -2,
            magnitude.rows & -2
        )
    );

    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;

    cv::Mat q0(
        magnitude,
        cv::Rect(0, 0, cx, cy)
    );

    cv::Mat q1(
        magnitude,
        cv::Rect(cx, 0, cx, cy)
    );

    cv::Mat q2(
        magnitude,
        cv::Rect(0, cy, cx, cy)
    );

    cv::Mat q3(
        magnitude,
        cv::Rect(cx, cy, cx, cy)
    );

    cv::Mat temp;

    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);

    cv::normalize(
        magnitude,
        magnitude,
        0,
        255,
        cv::NORM_MINMAX
    );

    magnitude.convertTo(
        magnitude,
        CV_8U
    );

    return magnitude;
}

int main()
{
    cv::Mat image = cv::imread(
        "C:/Users/USER/Desktop/GoruntuIsleme/Hafta7/Gorev07/resim.jpeg",
        cv::IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        std::cout
            << "Goruntu okunamadi"
            << std::endl;

        return -1;
    }

    cv::Mat harmonikMatris =
        ikiBoyutluDFT(image);

    cv::Mat harmonikGoruntu =
        harmonikGoster(harmonikMatris);

    cv::imshow(
        "Orijinal Goruntu",
        image
    );

    cv::imshow(
        "Harmonik Matrisi",
        harmonikGoruntu
    );

    cv::waitKey(0);

    return 0;
}
