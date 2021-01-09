// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "conio.h"
#include <queue>
#include <random>
using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}
 
void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}



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
	for (int i = 0; i<hist_cols; i++)
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
			for (int j = 0; j < src.cols; j++){
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

Mat_<float> convolutionWithoutNormalization(Mat_<uchar> src, int k, int k2, float * vals) {

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
				hist[(img(i,j)/m)]++;
		}

	int M = img.rows * img.cols;

	for (int k = 0; k <= 255; k++)
		pdf[k] = (float)hist[k] / M;

}

Mat_ < uchar> applyThreshold(Mat_<uchar> img, long double threshold) {

	Mat_ < uchar> result(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) > threshold )
				result(i, j) = 255;
			else 
				result(i, j) = 0;
		}

	/*imshow("Result threshold", result);
	waitKey(0);*/
	return result;
}

Mat_ < uchar> canny(Mat_<uchar> src) {
	
	/*imshow("original image", src);
	waitKey(0);*/

	//todo - separate sobel, prewitt in different methods and then use them here if necessary

	// Sobel dx
	Mat_<float> dxSobel = Mat(src.rows, src.cols, CV_32FC1);
	float * vals = new float[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dxSobel = convolutionWithoutNormalization(src,1,1,vals);
	/*imshow("dx-Sobel", abs(dxSobel) / 255);
	waitKey(0);*/

	// Sobel dy
	Mat_<float> dySobel = Mat(src.rows, src.cols, CV_32FC1);
	float * valsSecond = new float[9]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	dySobel = convolutionWithoutNormalization(src, 1, 1, valsSecond);
	/*imshow("dy-Sobel", abs(dySobel) / 255);
	waitKey(0);*/
	
	

	Mat_<float> magh = Mat(src.rows, src.cols, CV_32FC1);
	Mat_<float> angles = Mat(src.rows, src.cols, CV_32FC1);

	for (int i = 0; i < dxSobel.rows; i++)
		for (int j = 0; j < dxSobel.cols; j++) {
			magh(i, j) = sqrt(dxSobel(i,j) * dxSobel(i,j) + dySobel(i,j) * dySobel(i,j));
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

			beta = (int) (((angles(i, j) * 8) / (2 * CV_PI)) + 0.5) % 8;  //floor( ((angles(i, j) * 8) / (2 * CV_PI)) + 0.5) % 8;
			beta2 = (int) (beta + 4) % 8;

			if (i + dirx[beta] < magh.rows && i + dirx[beta] >= 0 && j + diry[beta] < magh.cols && j + diry[beta] >= 0
				&& i + dirx[beta2] < magh.rows && i + dirx[beta2] >= 0 && j + diry[beta2] < magh.cols && j + diry[beta2] >= 0) {
				if ((magh(i + dirx[beta], j + diry[beta]) >= magh(i,j) )
					|| (magh(i + dirx[beta2], j + diry[beta2]) >= magh(i,j) ))
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
			if (thinnedMagh(i, j) > t1 && thinnedMagh(i, j) < t2)
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

	// Get the histogram
	long double histogram[256];

	// initialize all intensity values to 0
	for (int i = 0; i < 256; i++)
		histogram[i] = 0;

	// calculate the no of pixels for each intensity values
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
	for (int i = 0;i < 255; i++) {
		if (maxi < Inter_class_variance[i]) {
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}

	//cout << "Otsu's algorithm implementation thresholding result: " << bin_mids[getmax];
	return bin_mids[getmax];
}

Mat_<uchar> Prewitt(Mat_<uchar> src) {
	// Prewitt dx
	Mat_<float> dxPrewitt = Mat(src.rows, src.cols, CV_32FC1);
	float* valsPrewitt = new float[9]{ -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	dxPrewitt = convolutionWithoutNormalization(src, 1, 1, valsPrewitt);
	/*imshow("dx-Prewitt", abs(dxPrewitt) / 255);
	waitKey(0);*/

	// Prewitt dy
	Mat_<float> dyPrewitt = Mat(src.rows, src.cols, CV_32FC1);
	float* valsSecondPrewitt = new float[9]{ 1, 1, 1, 0, 0, 0, -1, -1, -1 };
	dyPrewitt = convolutionWithoutNormalization(src, 1, 1, valsSecondPrewitt);
	/*imshow("dy-Prewitt", abs(dyPrewitt) / 255);
	waitKey(0);*/
	return dxPrewitt;
}

Mat_<uchar> Sobel(Mat_ < uchar> src) {
	// Sobel dx
	Mat_<float> dxSobel = Mat(src.rows, src.cols, CV_32FC1);
	float* vals = new float[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dxSobel = convolutionWithoutNormalization(src, 1, 1, vals);
	/*imshow("dx-Sobel", abs(dxSobel) / 255);
	waitKey(0);*/

	// Sobel dy
	Mat_<float> dySobel = Mat(src.rows, src.cols, CV_32FC1);
	float* valsSecond = new float[9]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	dySobel = convolutionWithoutNormalization(src, 1, 1, valsSecond);
	/*imshow("dy-Sobel", abs(dySobel) / 255);
	waitKey(0);*/
	return dxSobel;
}

int testVideoSequenceAll()
{
		VideoCapture cap(0);
		// Check if camera opened successfully
		
		if (!cap.isOpened())
		{
				cout << "Error opening video stream" << endl;
				return -1;
		}
	
		int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
		int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
		Mat_<uchar> edges;
		Mat_<uchar> edges_otsu;
		Mat_<uchar> edges_Prewitt;
		Mat_<uchar> edges_Sobel;
		int c = 0;
		while (1)
		{
				Mat frame;
				Mat grayFrame;
				cap >> frame;
				if (frame.empty())
					break;
				cvtColor(frame, grayFrame, CV_BGR2GRAY);
				edges=canny(grayFrame);
				long double thres = otsu(grayFrame);
				edges_otsu=applyThreshold(grayFrame, thres);
				edges_Prewitt = Prewitt(grayFrame);
				edges_Sobel = Sobel(grayFrame);

				//video.write(edges);
				//imshow("Frame", edges);
				
				// Display the resulting frame    	
				imshow("Canny", edges);
				imshow("Otsu", edges_otsu);
				imshow("Prewitt", edges_Prewitt);
				imshow("Sobel", edges_Sobel);
			

				// Press  ESC on keyboard to  exit

				char c = (char)waitKey(1);
				//c++;
				if (c == 27)
					break;
			
		}

		// When everything done, release the video capture and write object
		cap.release();
		video.release();
		// Closes all the windows
		destroyAllWindows();
		return 0;
	//VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	////VideoCapture cap(0);	// live video from web cam
	//if (!cap.isOpened()) {
	//	printf("Cannot open video capture device.\n");
	//	waitKey(0);
	//	return;
	//}

	//int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	//int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	//VideoWriter video("canny.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

	//Mat edges;
	//Mat frame;
	//char c;

	//while (cap.read(frame))
	//{
	//	Mat grayFrame;
	//	cap >> frame;
	//	cvtColor(frame, grayFrame, CV_BGR2GRAY);
	//	//Canny(grayFrame,edges,40,100,3);
	//	canny(grayFrame,edges);
	//	/*imshow("source", frame);
	//	imshow("gray", grayFrame);*/
	//	//imshow("edges", edges);

	//	

	//	video.write(edges);

	//	/*c = cvWaitKey(0);*/  // waits a key press to advance to the next frame
	//	//if (c == 27) {
	//	//	// press ESC to exit
	//	//	printf("ESC pressed - capture finished\n");
	//	//	break;  //ESC pressed
	//	//};
	//}
	//video.release();

	//todo save the frames to have a full processed & saved video
}



int testVideoSequenceCanny()
{
	VideoCapture cap(0);
	// Check if camera opened successfully

	if (!cap.isOpened())
	{
		cout << "Error opening video stream" << endl;
		return -1;
	}

	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
	Mat_<uchar> edges;
	Mat_<uchar> edgesC;

	int c = 0;
	while (1)
	{
		Mat frame;
		Mat grayFrame;
		cap >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		edges = canny(grayFrame);
		
		Canny(grayFrame, edgesC, 40, 100, 3);
		
		// Display the resulting frame    	
		imshow("Our Canny", edges);
		imshow("OpenCV Canny", edgesC);
		


		// Press  ESC on keyboard to  exit

		char c = (char)waitKey(1);
		//c++;
		if (c == 27)
			break;

	}

	// When everything done, release the video capture and write object
	cap.release();
	video.release();
	// Closes all the windows
	destroyAllWindows();
	return 0;
	
}


int testVideoSequenceOtsu()
{
	VideoCapture cap(0);
	// Check if camera opened successfully

	if (!cap.isOpened())
	{
		cout << "Error opening video stream" << endl;
		return -1;
	}

	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));
	Mat_<uchar> edges;
	Mat_<uchar> edgesO;

	int c = 0;
	while (1)
	{
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

		// Display the resulting frame    	
		imshow("Our Otsu", edges);
		imshow("OpenCV Otsu", edgesO);


		char c = (char)waitKey(1);
		//c++;
		if (c == 27)
			break;
	}

	// When everything done, release the video capture and write object
	cap.release();
	video.release();
	// Closes all the windows
	destroyAllWindows();
	return 0;

}

int main()
{
	int op;
	Mat_<uchar> img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/shapes.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/portrait_Salt&Pepper2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/portrait_Gauss1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat_<uchar> img = imread("Images/portrait_Gauss2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int * hist;
	float * pdf;
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
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Canny edge detection\n");
		printf(" 4 - Edges in a video sequence\n");
		printf(" 5 - Histogram \n");
		printf(" 6 - Canny \n");
		printf(" 7 - Otsu open cv \n");
		printf(" 8 - Otsu threshold computed \n");
		printf(" 9 - Our Canny vs Canny openCV \n");
		printf(" 10 - Our Sobel vs Sobel openCV \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);

		switch (op)
		{
			case 1:
			{
				testOpenImage();
				break;
			}
			case 2:
			{
				testOpenImagesFld();
				break;
			}
			case 3:
			{
				//canny open cv
				testCanny();
				break;
			}
			case 4:
			{
				testVideoSequenceAll();
				break;
			}
			case 5:
			{
				computeHistogram(img, hist, pdf, 1);
				showHistogram("Histogram", hist, 256, 300);
				waitKey(0);
				break;
			}
			case 6:
			{
				//canny computed
				
				/*canny(img);*/
				break;
			}
			case 7:
			{
				//open cv version of threshold
				Mat_<uchar> dst;
				double thresh = 0;
				double maxValue = 255;
				long double thres = cv::threshold(img, dst, thresh, maxValue, THRESH_OTSU);
				cout << "Otsu Threshold : " << thres << endl;
				imshow("Otsu result OpenCV", dst);
				waitKey(0);
				break;
			}
			case 8:
			{
				//otsu threshold computed
				long double thres = otsu(img);
				applyThreshold(img, thres);
				break;
			}
			case 9:
			{
				testVideoSequenceCanny();
				break;
			}
			case 10:
			{
				testVideoSequenceOtsu();
				break;
			}
		}
	}
	while (op!=0);
	return 0;
}