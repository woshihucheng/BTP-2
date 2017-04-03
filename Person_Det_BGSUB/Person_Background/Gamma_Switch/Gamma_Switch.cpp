/*This program performs backgrond segmentation using simple subtraction and blurring*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	if (argc != 7)
	{
		//cout << argc << endl;
		cout << " Usage: Gamma_Switch Video_Directory Threshold_value Iteration GAMMA_SWITCH minArea randseed" << endl;
		return -1;
	}
	VideoCapture cap(argv[1]); // open the video file

	if (!cap.isOpened())  // check if we succeeded
		return -1;
	Mat frame, ThreshFrame;
	cuda::GpuMat src, dst, backgrnd, framediff; // Cuda frames input and output
	namedWindow("frame", 1);
	Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(CV_8U, -1, cv::Size(21, 21), 1);//Create gaussian filter pointer
	
	/*Morpholocial Dilation Parameters*/
	Mat kernel = getStructuringElement(MORPH_RECT, Size(8 * 2 + 1, 8 * 2 + 1), Point(8, 8));
	Ptr<cuda::Filter> dil_Filt = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, Mat(), Point(-1, -1), atoi(argv[3]));
	
	RNG rng(atoi(argv[6]));
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	long long int frm_cnt = 0;
	int64 t0 = 0;

	bool gm = (bool)atoi(argv[4]);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			cout << "Empty frame" << endl;
			break;
		}
		frm_cnt++;
		src.upload(frame);//Uploading frame to the device memory		
		if (gm)
		{
			cuda::gammaCorrection(src, src);
			src.download(frame);
		}
		cuda::cvtColor(src, src, COLOR_BGR2GRAY);//Convert to gray scale
		gauss->apply(src, src);//Apply gausssian filter(blurring)
		if (backgrnd.empty())
		{
			backgrnd = src.clone();//To copy the data in backgrnd frame
			cout << "First frame" << endl;
			t0 = cv::getTickCount();
			continue;//No processing for the first frame
		}

		cuda::absdiff(src, backgrnd, framediff);
		cuda::threshold(framediff, framediff, atoi(argv[2]), 255, cv::THRESH_BINARY);
		dil_Filt->apply(framediff, framediff);
		framediff.download(ThreshFrame);

		findContours(ThreshFrame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}
		for (int i = 0; i< contours.size(); i++)
		{

			if (boundRect[i].width * boundRect[i].height > atoi(argv[5]))
			{
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			}
		}
		
		imshow("frame", frame);
		//imshow("Theshframe", ThreshFrame);
		if (waitKey(1) >= 0)
			break;
	}
	int64 t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "FPS" << (double)frm_cnt / secs;
	return 0;
}