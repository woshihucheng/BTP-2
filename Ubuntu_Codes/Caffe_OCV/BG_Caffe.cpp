#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include<opencv2/cudalegacy.hpp>
#include<opencv2/cudabgsegm.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include "Caffe_Helper.h"
using namespace cv;
using namespace cv::cuda;
using namespace std;


string ToString(int val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

int main(int argc, char** argv)
{
	if (argc != 10)
	{
		cout << " Usage: " << argv[0] << " Video_Directory minArea randseed win_width win_height" <<
		" deploy.prototxt network.caffemodel mean.binaryproto labels.txt" <<endl;
		return -1;
	}
	
	/*Initialization of model*/
	//::google::InitGoogleLogging(argv[0]);
	const string model_file   = argv[6];
    const string trained_file = argv[7];
  	const string mean_file    = argv[8];
	const string label_file = argv[9];
	VideoCapture cap(argv[1]); // open the video file
	Classifier classifier(model_file, trained_file, mean_file, label_file);
	cout << "a" <<  endl;
	cv::Size winSz(atoi(argv[4]), atoi(argv[5]));
	if (!cap.isOpened())  // check whether video capture object is opened or not
		return -1;
	Mat h_frame, h_foreground, h_fgmask;//Host frames
	cuda::GpuMat d_foreground, d_fgmask; //Device(Gpu) images

										 /*Imshow windows initialization*/
	namedWindow("Foreground", 1);
	//namedWindow("h_ROI", 1);
	namedWindow("a", 1);

	Ptr<BackgroundSubtractor> fgd = cuda::createBackgroundSubtractorFGD();
	Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
	Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractor> gmg = cuda::createBackgroundSubtractorGMG(40);
	vector<Ptr<BackgroundSubtractor> > v_models;
	v_models.push_back(fgd);
	v_models.push_back(mog);
	v_models.push_back(mog2);
	v_models.push_back(gmg);
	/*Initial run for FGD*/
	cap >> h_frame;
	cuda::GpuMat d_frame(h_frame);
	gmg->apply(d_frame, d_fgmask);

	RNG rng(atoi(argv[3]));
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	int mod = 3;
	long long int frm_cnt = 0;

	

	cv::Mat h_ROI;
	int img_no=0;
	int64 t0 = cv::getTickCount();//FPS count start
	for (;;)
	{
		cap >> h_frame; // get a new frame from camera
		if (h_frame.empty())
		{
			cout << "Empty frame" << endl;
			break;
		}
		frm_cnt++;
		//cout << "bcc";
		d_frame.upload(h_frame);//Uploading frame to the device memory
		if (mod == 0)
			v_models[mod]->apply(d_frame, d_fgmask, 0.01);
		else
			v_models[mod]->apply(d_frame, d_fgmask);//Background subtraction
		d_foreground.create(d_frame.size(), d_frame.type());
		d_foreground.setTo(Scalar::all(0));
		d_frame.copyTo(d_foreground, d_fgmask);
		d_foreground.download(h_foreground);
		d_fgmask.download(h_fgmask);
		findContours(h_fgmask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			//cout << "aa";
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}
		for (int i = 0; i< contours.size(); i++)
		{

			if (boundRect[i].width * boundRect[i].height > atoi(argv[2]))
			{
				
				rectangle(h_frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);				
				Mat ROI(h_frame, boundRect[i]);
				//cout << boundRect[i].height << " " << boundRect[i].width << endl;
				resize(ROI, ROI, cv::Size(64, 64), 0, 0, INTER_AREA);
				//cv::cuda::GpuMat d_ROI(ROI);
								
				std::vector<Prediction> predictions = classifier.Classify(ROI,1);
				for (size_t j = 0; j < predictions.size(); ++j)
				{
    				Prediction p = predictions[j];
    				std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              		<< p.first << "\"" << std::endl;
              		if(p.first == "Bag")
						imwrite("./Bag/"+ToString(img_no++)+".jpg",ROI);
					else if(p.first == "NoBag")
						imwrite("./NoBag/"+ToString(img_no++)+".jpg",ROI);
					else
						imwrite("./Background/"+ToString(img_no++)+".jpg",ROI);

				}
				
				imshow("a", ROI);
				

			}
		}
		imshow("Foreground", h_frame);
		//if(!h_ROI.empty())
		//	imshow("h_ROI", h_ROI);
		switch ((char)waitKey(1))
		{
		case '0':
			mod = 0;
			break;
		case '1':
			mod = 1;
			break;
		case '2':
			mod = 2;
			break;
		case '3':
			mod = 3;
			break;
		case 'e':
			mod = 4;
			break;
		default:
			break;
		}
		if (mod == 4)
			break;
	}
	int64 t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "FPS" << (double)frm_cnt / secs;
	return 0;
}
