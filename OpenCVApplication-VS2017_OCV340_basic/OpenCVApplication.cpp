// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "conio.h"
#include <queue>
#include <random>
using namespace std;

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void convertColorToGrayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname);
		Mat_<uchar> destination(src.rows, src.cols);

		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				uchar b = src(i, j)[0];
				uchar g = src(i, j)[1];
				uchar r = src(i, j)[2];
				destination(i, j) = (r + g + b) / 3;  // cand aduni 3 charuri, se converteste automat la int // nu ramane suma in uchar si face modulo
			}


		imshow("source - color", src);
		imshow("destination - gray", destination);
		waitKey();
	}
}

Mat_<float> convolutionWithoutNormalization(Mat_<uchar> src, int k, int k2, float* vals) {

	Mat_<float> H(2 * k + 1, 2 * k2 + 1, vals);
	Mat_<float> dest(src.rows, src.cols);

	int  w = 2 * k + 1;
	int w1 = 2 * k2 + 1;
	float val = 0.0f;

	for (int i = k; i < src.rows - k; i++)
		for (int j = k2; j < src.cols - k2; j++) {

			val = 0.0f;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w1; v++) {
					val += H(u, v) * src(i + u - k, j + v - k2);
				}
			}

			dest(i, j) = val;
		}

	return dest;

}

void computeHistogram(Mat_<uchar> img, int* hist, float* pdf, int m) {

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			hist[(img(i, j) / m)]++;
		}

	int M = img.rows * img.cols;

	for (int k = 0; k <= 255; k++)
		pdf[k] = (float)hist[k] / M;

}

Mat_ < uchar> applyThreshold(Mat_<uchar> img, long double threshold) {

	Mat_ < uchar> result(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) > threshold)
				result(i, j) = 255;
			else
				result(i, j) = 0;
		}

	/*imshow("Result threshold", result);
	waitKey(0);*/
	return result;
}

Mat_ < uchar> canny(Mat_<uchar> src) {


	// Sobel dx
	Mat_<float> dxSobel = Mat(src.rows, src.cols, CV_32FC1);
	float* vals = new float[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dxSobel = convolutionWithoutNormalization(src, 1, 1, vals);
	

	// Sobel dy
	Mat_<float> dySobel = Mat(src.rows, src.cols, CV_32FC1);
	float* valsSecond = new float[9]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	dySobel = convolutionWithoutNormalization(src, 1, 1, valsSecond);
	

	Mat_<float> magh = Mat(src.rows, src.cols, CV_32FC1);
	Mat_<float> angles = Mat(src.rows, src.cols, CV_32FC1);

	for (int i = 0; i < dxSobel.rows; i++)
		for (int j = 0; j < dxSobel.cols; j++) {
			magh(i, j) = sqrt(dxSobel(i, j) * dxSobel(i, j) + dySobel(i, j) * dySobel(i, j));
			angles(i, j) = atan2(dySobel(i, j), dxSobel(i, j));
			if (angles(i, j) < 0)
				angles(i, j) += 2 * CV_PI;
		}

	/*imshow("magh", magh);
	waitKey(0);*/

	int dirx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int diry[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Mat_<float> thinnedMagh = magh.clone();
	int beta;
	int beta2;
	for (int i = 0; i < magh.rows; i++)
		for (int j = 0; j < magh.cols; j++) {

			beta = (int)(((angles(i, j) * 8) / (2 * CV_PI)) + 0.5) % 8;  //floor( ((angles(i, j) * 8) / (2 * CV_PI)) + 0.5) % 8;
			beta2 = (int)(beta + 4) % 8;

			if (i + dirx[beta] < magh.rows && i + dirx[beta] >= 0 && j + diry[beta] < magh.cols && j + diry[beta] >= 0
				&& i + dirx[beta2] < magh.rows && i + dirx[beta2] >= 0 && j + diry[beta2] < magh.cols && j + diry[beta2] >= 0) {
				if ((magh(i + dirx[beta], j + diry[beta]) >= magh(i, j))
					|| (magh(i + dirx[beta2], j + diry[beta2]) >= magh(i, j)))
					thinnedMagh(i, j) = 0;
			}
		}

	/*imshow("thinned", abs(thinnedMagh) / 255);
	waitKey(0);*/

	Mat_ < uchar> result(thinnedMagh.rows, thinnedMagh.cols);
	/*int t1 = 150;
	int t2 = 200;*/
	
	int t1 = 40;
	int t2 = 100;


	for (int i = 0; i < thinnedMagh.rows; i++)
		for (int j = 0; j < thinnedMagh.cols; j++) {
			if (thinnedMagh(i, j) > t1&& thinnedMagh(i, j) < t2)
				result(i, j) = 128;
			else if (thinnedMagh(i, j) > t2)
				result(i, j) = 255;
			else if (thinnedMagh(i, j) < t1)
				result(i, j) = 0;
		}

	/*imshow("3way threshold", result);
	waitKey(0);*/

	std::queue<Point2i> Q;
	Point2i elem;

	for (int i = 0; i < result.rows; i++)
		for (int j = 0; j < result.cols; j++) {

			if (result.at<uchar>(i, j) == 255)
				Q.push({ i, j });

			while (!(Q.empty())) {
				elem = Q.front();
				Q.pop();

				for (int k = 0; k < 8; k++) {
					if (i + dirx[k] < result.rows && i + dirx[k] >= 0 && j + diry[k] < result.cols && j + diry[k] >= 0) {
						if (result.at<uchar>(elem.x + dirx[k], elem.y + diry[k]) == 128) {
							result.at<uchar>(elem.x + dirx[k], elem.y + diry[k]) = 255;
							Q.push({ elem.x + dirx[k], elem.y + diry[k] });
						}
					}

				}
			}
		}

	// if the weak is in air, make it black
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (result.at<uchar>(i, j) == 128) {
				result.at<uchar>(i, j) = 0;
			}
		}

	/*imshow("result edge linking", result);
	waitKey(0);*/
	return result;
}

long double otsu(Mat_<uchar> src) {
	int bins_num = 256;

	// Get the histogram and initialize all values to 0
	long double histogram[256];
	for (int i = 0; i < 256; i++)
		histogram[i] = 0;
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			histogram[(int)src.at<uchar>(y, x)]++;

	// Calculate the bin_edges
	long double bin_edges[256];
	bin_edges[0] = 0.0;
	long double increment = 0.99609375;
	for (int i = 1; i < 256; i++)
		bin_edges[i] = bin_edges[i - 1] + increment;

	// Calculate bin_mids
	long double bin_mids[256];
	for (int i = 0; i < 256; i++)
		bin_mids[i] = (bin_edges[i] + bin_edges[i + 1]) / 2;

	// Iterate over all thresholds (indices) and get the probabilities weight1, weight2
	long double weight1[256];
	weight1[0] = histogram[0];
	for (int i = 1; i < 256; i++)
		weight1[i] = histogram[i] + weight1[i - 1];

	int total_sum = 0;
	for (int i = 0; i < 256; i++)
		total_sum = total_sum + histogram[i];
	long double weight2[256];
	weight2[0] = total_sum;
	for (int i = 1; i < 256; i++)
		weight2[i] = weight2[i - 1] - histogram[i - 1];

	// Calculate the class means: mean1 and mean2
	long double histogram_bin_mids[256];
	for (int i = 0; i < 256; i++)
		histogram_bin_mids[i] = histogram[i] * bin_mids[i];

	long double cumsum_mean1[256];
	cumsum_mean1[0] = histogram_bin_mids[0];
	for (int i = 1; i < 256; i++)
		cumsum_mean1[i] = cumsum_mean1[i - 1] + histogram_bin_mids[i];

	long double cumsum_mean2[256];
	cumsum_mean2[0] = histogram_bin_mids[255];
	for (int i = 1, j = 254; i < 256 && j >= 0; i++, j--)
		cumsum_mean2[i] = cumsum_mean2[i - 1] + histogram_bin_mids[j];

	long double mean1[256];
	for (int i = 0; i < 256; i++)
		mean1[i] = cumsum_mean1[i] / weight1[i];

	long double mean2[256];
	for (int i = 0, j = 255; i < 256 && j >= 0; i++, j--)
		mean2[j] = cumsum_mean2[i] / weight2[j];

	// Calculate Inter_class_variance
	long double Inter_class_variance[255];
	long double dnum = 10000000000;
	for (int i = 0; i < 255; i++)
		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i + 1])) / dnum) * (mean1[i] - mean2[i + 1]);


	// Maximize interclass variance
	long double maxi = 0;
	int getmax = 0;
	for (int i = 0; i < 255; i++) {
		if (maxi < Inter_class_variance[i]) {
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}

	//cout << "Otsu's algorithm thresholding result: " << bin_mids[getmax];
	return bin_mids[getmax];
}

Mat_<uchar> prewitt(Mat_<uchar> src) {

	// Prewitt dx
	Mat_<float> dxPrewitt = Mat(src.rows, src.cols, CV_32FC1);
	float* valsPrewitt = new float[9]{ -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	dxPrewitt = convolutionWithoutNormalization(src, 1, 1, valsPrewitt);

	// Prewitt dy
	Mat_<float> dyPrewitt = Mat(src.rows, src.cols, CV_32FC1);
	float* valsSecondPrewitt = new float[9]{ 1, 1, 1, 0, 0, 0, -1, -1, -1 };
	dyPrewitt = convolutionWithoutNormalization(src, 1, 1, valsSecondPrewitt);
	Mat_<float> magh = Mat(src.rows, src.cols, CV_32FC1);


	for (int i = 0; i < dxPrewitt.rows; i++)
		for (int j = 0; j < dxPrewitt.cols; j++) {
			magh(i, j) = sqrt(dxPrewitt(i, j) * dxPrewitt(i, j) + dyPrewitt(i, j) * dyPrewitt(i, j));
		}

	return magh;
}

Mat_<uchar> sobel(Mat_ < uchar> src) {

	// Sobel dx
	Mat_<float> dxSobel = Mat(src.rows, src.cols, CV_32FC1);
	float* vals = new float[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dxSobel = convolutionWithoutNormalization(src, 1, 1, vals);

	// Sobel dy
	Mat_<float> dySobel = Mat(src.rows, src.cols, CV_32FC1);
	float* valsSecond = new float[9]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	dySobel = convolutionWithoutNormalization(src, 1, 1, valsSecond);
	Mat_<float> magh = Mat(src.rows, src.cols, CV_32FC1);
	

	for (int i = 0; i < dxSobel.rows; i++)
		for (int j = 0; j < dxSobel.cols; j++) {
			magh(i, j) = sqrt(dxSobel(i, j) * dxSobel(i, j) + dySobel(i, j) * dySobel(i, j));
		}

	return magh;
}

int testVideoSequenceAll()
{
	VideoCapture cap(0);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream" << endl;
		return -1;
	}

	//int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	//int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	//VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

	Mat_<uchar> edges;
	Mat_<uchar> edges_otsu;
	Mat_<uchar> edges_Prewitt;
	Mat_<uchar> edges_Sobel;
	int c = 0;

	while (1) {
		Mat frame;
		Mat grayFrame;

		cap >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		edges = canny(grayFrame);
		long double thres = otsu(grayFrame);
		edges_otsu = applyThreshold(grayFrame, thres);
		edges_Prewitt = prewitt(grayFrame);
		edges_Sobel = sobel(grayFrame);

		//video.write(edges);
		//imshow("Frame", edges);

		// Display the resulting frame    	
		imshow("Canny", edges);
		imshow("Otsu", edges_otsu);
		imshow("Prewitt", edges_Prewitt);
		imshow("Sobel", edges_Sobel);


		// Press  ESC on keyboard to  exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;

	}

	cap.release();
	//video.release();

	destroyAllWindows();
	return 0;
}

int testVideoSequenceCanny()
{
	VideoCapture cap(0);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream" << endl;
		return -1;
	}

	Mat_<uchar> edges;
	Mat_<uchar> edgesC;

	int c = 0;
	while (1) {
		Mat frame;
		Mat grayFrame;
		cap >> frame;

		if (frame.empty())
			break;

		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		edges = canny(grayFrame);
		Canny(grayFrame, edgesC, 40, 100, 3);

		imshow("Our Canny", edges);
		imshow("OpenCV Canny", edgesC);
	

		// Press  ESC on keyboard to  exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;

	}

	cap.release();
	destroyAllWindows();
	return 0;
}

int testVideoSequenceOtsu()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "Error opening video stream" << endl;
		return -1;
	}

	Mat_<uchar> edges;
	Mat_<uchar> edgesO;

	int c = 0;
	while (1) {
		Mat frame;
		Mat grayFrame;

		cap >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, grayFrame, CV_BGR2GRAY);

		long double thres = otsu(grayFrame);
		edges = applyThreshold(grayFrame, thres);

		double thresh = 0;
		double maxValue = 255;
		long double threst = cv::threshold(grayFrame, edgesO, thresh, maxValue, THRESH_OTSU);

		imshow("Our Otsu", edges);
		imshow("OpenCV Otsu", edgesO);

		// Press  ESC on keyboard to  exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;
	}

	cap.release();
	destroyAllWindows();
	return 0;

}

int main()
{
	int op;
	//Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/basket.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/bear.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//1)Mat_<uchar> img = imread("Images/bear_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> img = imread("Images/brush.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/elephants.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/elephants_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/goat_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/golfcart.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/lions.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/rino.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/turtle.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	

	int* hist;
	float* pdf;
	hist = new int[256];
	pdf = new float[256];
	int m = 256;

	for (int i = 0; i <= 255; i++) {
		hist[i] = 0;
		pdf[i] = 0.0f;
	}

	do
	{
		//system("cls");
		destroyAllWindows();
		printf("Menu: \n");
		printf(" 1 - Edges in a video sequence - all edge detection algorithms \n");
		printf(" 2 - Edges in a video sequence - our canny vs OpenCV canny \n");
		printf(" 3 - Edges in a video sequence - our otsu vc OpenCV otsu \n");
		printf(" 4 - Histogram \n");
		printf(" 5 - Canny - on image \n");
		printf(" 6 - Otsu - on image \n");
		printf(" 7 - Sobel - on image \n");
		printf(" 8 - Prewitt - on image \n");
		printf(" 9 - Our Canny vs Canny OpenCV - on image \n");
		printf(" 10 - Our Otsu vs Otsu OpenCV - on image \n");
		printf(" 11 - All edge detection algorithms - on image \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op)
		{
		case 1:
		{
			testVideoSequenceAll();
			break;
		}
		case 2:
		{
			testVideoSequenceCanny();
			break;
		}
		case 3:
		{
			testVideoSequenceOtsu();
			break;
		}
		case 4:
		{
			computeHistogram(img, hist, pdf, 1);
			showHistogram("Histogram", hist, 256, 300);
			waitKey(0);
			break;
		}
		case 5:
		{
			Mat_<uchar> cannyEdges;
			cannyEdges = canny(img);
			imshow("Canny Output", cannyEdges);
			waitKey(0);
			break;
		}
		case 6:
		{
			Mat_<uchar> otsuEdges;
			long double thres = otsu(img);
			otsuEdges = applyThreshold(img, thres);
			imshow("Otsu Output", otsuEdges);
			waitKey(0);
			break;
		}
		case 7:
		{
			Mat_<uchar> sobelEdges;
			sobelEdges = sobel(img);
			imshow("Sobel Output", sobelEdges);
			waitKey(0);
			break;
		}
		case 8:
		{
			Mat_<uchar> prewittEdges;
			prewittEdges = prewitt(img);
			imshow("Prewitt Output", prewittEdges);
			waitKey(0);
			break;
		}
		case 9:
		{
			//our canny
			Mat_<uchar> cannyEdges;
			cannyEdges = canny(img);

			//opencv canny
			Mat_<uchar> cannyOpenCvEdges;
			Canny(img, cannyOpenCvEdges, 40, 100, 3);

			imshow("Canny", cannyEdges);
			imshow("Canny OpenCV", cannyOpenCvEdges);
			waitKey(0);
			break;
		}
		case 10:
		{
			//our otsu
			Mat_<uchar> otsuEdges;
			long double thres = otsu(img);
			otsuEdges = applyThreshold(img, thres);

			//opencv otsu
			Mat_<uchar> otsuOpenCvEdges;
			double thresh = 0;
			double maxValue = 255;
			long double threst = cv::threshold(img, otsuOpenCvEdges, thresh, maxValue, THRESH_OTSU);

			imshow("Otsu", otsuEdges);
			imshow("Otsu OpenCV", otsuOpenCvEdges);
			waitKey(0);
			break;
		}
		case 11:
		{
			//all applied on photos
			Mat_<uchar> cannyEdges;
			Mat_<uchar> otsuEdges;
			Mat_<uchar> prewittEdges;
			Mat_<uchar> sobelEdges;

			cannyEdges = canny(img);

			long double thres = otsu(img);
			otsuEdges = applyThreshold(img, thres);

			prewittEdges = prewitt(img);

			sobelEdges = sobel(img);

			imshow("Canny", cannyEdges);
			imshow("Otsu", otsuEdges);
			imshow("Prewitt", prewittEdges);
			imshow("Sobel", sobelEdges);
			waitKey(0);
			break;
		}
		}
	} while (op != 0);
	return 0;
}