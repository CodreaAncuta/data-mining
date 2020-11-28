#include "stdafx.h"
#include "common.h"

void ColorTo3matrices() {
	// Mat_<Vec3b> img(256, 256, CV_8UC3);

	Mat_<Vec3b> img = imread("Images/kids.bmp",
		CV_LOAD_IMAGE_COLOR);

	Mat_<uchar> M1 = Mat(img.rows, img.cols, CV_8UC1);
	Mat_<uchar> M2 = Mat(img.rows, img.cols, CV_8UC1);
	Mat_<uchar> M3 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			M1(i, j) = img(i, j)[0]; //b
			M2(i, j) = img(i, j)[1]; //g
			M3(i, j) = img(i, j)[2]; //r
		}

	imshow("blue", M1);
	imshow("green", M2);
	imshow("red", M3);
	waitKey();
}

void convertColorToGrayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar b = src.at<Vec3b>(i, j)[0];
				uchar g = src.at<Vec3b>(i, j)[1];
				uchar r = src.at<Vec3b>(i, j)[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void HSVfromRGB() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		float r, g, b, M, m, C;

		// Componentele de culoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				b = (float)v3[0] / 255;
				g = (float)v3[1] / 255;
				r = (float)v3[2] / 255;

				M = max(max(r, g), b);
				// std::cout << M << "\n";
				m = min(min(r, g), b);
				C = M - m;

				V.at<uchar>(i, j) = M; // value

				// saturation
				if (V.at<uchar>(i, j) != 0) {
					S.at<uchar>(i, j) = C / V.at<uchar>(i, j);
				}
				else S.at<uchar>(i, j) = 0;
				// hue
				if (C != 0) {
					if (M == r) H.at<uchar>(i, j) = 60 * (g - b) / C;
					if (M == g) H.at<uchar>(i, j) = 120 + 60 * (b - r) / C;
					if (M == b) H.at<uchar>(i, j) = 240 + 60 * (r - g) / C;
				}
				else H.at<uchar>(i, j) = 0;
				if (H.at<uchar>(i, j) < 0) H.at<uchar>(i, j) += 360;

			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		waitKey();
	}

}

void isInside(Mat_<uchar> img, int i, int j) {
	for (int x = 0; x < img.rows; x++)
		for (int y = 0; y < img.cols; y++) {
			if ((x == i) && (y == j))
				std::cout << "(i,j) is inside!";
		}
}