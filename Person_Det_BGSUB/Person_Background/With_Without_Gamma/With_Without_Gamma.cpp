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
	if (argc != 5)
	{
		//cout << argc << endl;
		cout << " Usage: Opencv_try Video_Directory Threshold_value Iteration 1/0" << endl;
		return -1;
	}
	VideoCapture cap(argv[1]); // open the video file

	if (!cap.isOpened())  // check if we succeeded
		return -1;
	Mat frame, ThreshFrame;
	Mat woframe, woThreshFrame;
	cuda::GpuMat src, dst, backgrnd, framediff; // Cuda frames input and output
	cuda::GpuMat wosrc, wodst, wobackgrnd, woframediff;
	namedWindow("frame", 1);
	namedWindow("Threshframe", 1);
	namedWindow("WithoutGammaframe", 1);
	long long int frm_cnt = 0;
	Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(CV_8U, -1, cv::Size(21, 21), 1);//Create gaussian filter pointer

																						 /*Morpholocial Dilation Parameters*/
	Mat kernel = getStructuringElement(MORPH_RECT, Size(8 * 2 + 1, 8 * 2 + 1), Point(8, 8));
	Ptr<cuda::Filter> dil_Filt = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, Mat(), Point(-1, -1), atoi(argv[3]));

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
		wosrc.upload(frame);
		if (gm)
		{
			cuda::gammaCorrection(src, src);
			src.download(frame);
		}
		cuda::cvtColor(src, src, COLOR_BGR2GRAY);//Convert to gray scale
		cuda::cvtColor(wosrc, wosrc, COLOR_BGR2GRAY);//Convert to gray scale
		gauss->apply(src, src);//Apply gausssian filter(blurring)
		gauss->apply(wosrc, wosrc);//Apply gausssian filter(blurring)
		if (backgrnd.empty())
		{
			backgrnd = src.clone();//To copy the data in backgrnd frame
			wobackgrnd = wosrc.clone();
			cout << "First frame" << endl;
			t0 = cv::getTickCount();
			continue;//No processing for the first frame
		}

		cuda::absdiff(src, backgrnd, framediff);
		cuda::absdiff(wosrc, wobackgrnd, woframediff);
		cuda::threshold(framediff, framediff, atoi(argv[2]), 255, cv::THRESH_BINARY);
		cuda::threshold(woframediff, woframediff, atoi(argv[2]), 255, cv::THRESH_BINARY);
		dil_Filt->apply(framediff, framediff);
		dil_Filt->apply(woframediff, woframediff);
		framediff.download(ThreshFrame);
		woframediff.download(woThreshFrame);
		imshow("frame", frame);
		imshow("Theshframe", ThreshFrame);
		imshow("WithoutGammaframe", woThreshFrame);
		//if (waitKey(1) >= 0) break;
		if (waitKey(1) >= 0)
			break;
	}
	int64 t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "FPS" << (double)frm_cnt / secs;
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}