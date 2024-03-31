#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
	cv::Mat image = cv::imread("C:/Users/IVAN/Documents/moscow.jpg");

	if (image.empty()) {
		std::cout << "Error file!" << std::endl;
		return -1;
	}

	cv::Mat gray(image.rows, image.cols, CV_8UC1);
	cv::Mat contour(image.rows, image.cols, CV_8UC1);
	cv::Mat negative(image.rows, image.cols, CV_8UC3);
	cv::Mat sepia(image.rows, image.cols, CV_8UC3);
	#pragma omp parallel sections num_threads(4)
	{
		#pragma omp section
		{
			for (int i = 0; i < image.rows; ++i) {
				for (int j = 0; j < image.cols; ++j) {
					cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
					int gray_value = 0.29 * pixel[2] + 0.58 * pixel[1] + 0.11 * pixel[0];
					gray.at<uchar>(i, j) = gray_value;
				}
			}
		}
		#pragma omp section
		{
			cv::Mat gray;
			cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			for (int i = 1; i < gray.rows - 1; i++) {
				for (int j = 1; j < gray.cols - 1; j++) {
					float gx = gray.at<uchar>(i + 1, j + 1) + 2 * gray.at<uchar>(i, j + 1) + gray.at<uchar>(i - 1, j + 1) - gray.at<uchar>(i + 1, j - 1) - 2 * gray.at<uchar>(i, j - 1) - gray.at<uchar>(i - 1, j - 1);
					float gy = gray.at<uchar>(i + 1, j + 1) + 2 * gray.at<uchar>(i + 1, j) + gray.at<uchar>(i + 1, j - 1) - gray.at<uchar>(i - 1, j - 1) - 2 * gray.at<uchar>(i - 1, j) - gray.at<uchar>(i - 1, j + 1);

					contour.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
				}
			}

		}
		#pragma omp section
		{
			for (int i = 0; i < image.rows; ++i) {
				for (int j = 0; j < image.cols; ++j) {
					cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
					negative.at<cv::Vec3b>(i, j) = cv::Vec3b(255 - pixel[0], 255 - pixel[1], 255 - pixel[2]);
				}
			}
		}
		#pragma omp section
		{
			for (int i = 0; i < image.rows; ++i) {
				for (int j = 0; j < image.cols; ++j) {

					cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
					int blue = pixel[0];
					int green = pixel[1];
					int red = pixel[2];
					int RED = (int)(0.393 * blue + 0.769 * green + 0.189 * red);
					int GREEN = (int)(0.349 * blue + 0.686 * green + 0.168 * red);
					int BLUE = (int)(0.272 * blue + 0.534 * green + 0.131 * red);

					sepia.at<cv::Vec3b>(i, j) = cv::Vec3b(std::min(BLUE, 255), std::min(GREEN, 255), std::min(RED, 255));
				}
			}
		}
	}

	imshow("original", image);
	imshow("gray", gray);
	imshow("contour", contour);
	imshow("negative", negative);
	imshow("sepia", sepia);

	cv::waitKey(0);

	return 0;
}