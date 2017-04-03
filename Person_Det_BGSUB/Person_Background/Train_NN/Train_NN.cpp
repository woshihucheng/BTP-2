#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/ml.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace cv::ml;
using namespace std;

void load_filenames(vector<string> &filenames, string text_file_dir)
{
	string line;
	ifstream fl(text_file_dir);
	if (fl.is_open())
	{
		while (getline(fl, line))
		{
			filenames.push_back(line);
		}
	}
	else
	{
		cout << "Error opening file. Make sure you have used \\\\ instead of \\" << endl;
	}
}


int main(int argc, char **argv)
{
	vector<string> pos_filenames;
	vector<string> neg_filenames;

	if (argc != 6)
	{
		cout << "Usage: " << argv[0] << " pos_images_text_file neg_images_text_file"
			<< "width height output_file" << endl;
		cout << "To generate text files use PETA_read.exe file" << endl;
		cout << "Example: PETA_read PETA dataset\\ accessoryHat.txt accessoryHat" << endl;
		return 1;
	}

	int64 t0 = cv::getTickCount();//Beginning of the program

	
	string pos_dir = argv[1];//Directory for positive images
	string neg_dir = argv[2];//Directory for positive images
	cv::Size sz(atoi(argv[3]), atoi(argv[4])); //Opencv supports only multiple of 8x8 winStride as of now


	load_filenames(pos_filenames, pos_dir);

	cv::Mat train_data(0,sz.width*sz.height, CV_32FC1);//HOG freatures Matrix where each row correspondes to a sample
	for (size_t i = 0; i < pos_filenames.size(); ++i)
	{
		Mat src = imread(pos_filenames[i]);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;
		resize(src, src, sz, 0, 0, INTER_AREA);
		//cv::Size tsz2(src.size());
		//cout << tsz2.width << " " << tsz2.height << endl;
		if(src.channels()>=3)
			cvtColor(src, src,CV_BGR2GRAY);
		src.convertTo(src, CV_32FC1);
		src = src.reshape(1, 1);
		//tsz2 = src.size();
		//cout << tsz2.width << " " << tsz2.height << endl;
		train_data.push_back(src.clone());
	}
	cv::Size tsz(train_data.size());
	cout << tsz.width << " " << tsz.height << endl;
	cout << "Reading positive images done" << endl;
	
	cv::Mat pos_lab(train_data.rows, 1, CV_32FC1, Scalar(1.0));//Matrix of 1s
															 //pos_lab = Scalar(1);

	load_filenames(neg_filenames, neg_dir);
	long int k = 0;
	
	for (size_t i = 0; i < neg_filenames.size(); ++i)
	{
		
		Mat src = imread(neg_filenames[i]);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;
		resize(src, src, sz, 0, 0, INTER_AREA);
		if (src.channels() >= 3)
			cvtColor(src, src, CV_BGR2GRAY);
		src.convertTo(src, CV_32FC1);
		cv::Mat src4 = src.reshape(0, 1);
		train_data.push_back(src4.clone());

		/*Sampling Each Negative*/
		srand((unsigned int)time(NULL));//Setting the seed

		Rect samp_ROI;
		samp_ROI.height = 10;
		samp_ROI.width = 10;
		samp_ROI.x = rand() % (sz.width - samp_ROI.width-1);
		samp_ROI.y = rand() % (sz.height - samp_ROI.height-1);
		cv::Mat src2(src, samp_ROI);
		resize(src2, src2, sz, 0, 0, INTER_AREA);
		if (src2.channels() >= 3)
			cvtColor(src2, src2, CV_BGR2GRAY);
		src2.convertTo(src2, CV_32FC1);
		cv::Mat src3 = src2.reshape(0, 1);
		train_data.push_back(src3.clone());
		k += 2;
	}
	tsz = (train_data.size());
	cout << tsz.width << " " << tsz.height << endl;
	cout << "Reading files done" << endl;
	cout << "Number of positive sameples=" << pos_filenames.size() << endl;
	cout << "Number of negative sameples=" << k << endl;
	cv::Mat neg_lab(k, 1, CV_32FC1, Scalar(-1.0));//Matrix of -1s
	cv::Mat train_lab;
	cv::vconcat(pos_lab, neg_lab, train_lab);
	cout << "Generating Labels done" << endl;
	
	static Ptr<TrainData>  td = cv::ml::TrainData::create(train_data, ROW_SAMPLE, train_lab);
	td->setTrainTestSplitRatio(0.7, true);

	cout << "Generating train data pointer done" << endl;

	/*Training ANN*/
	cv::Ptr<ANN_MLP> model = cv::ml::ANN_MLP::create();
	cv::Mat tmpL(4, 1, CV_32SC1);
	tmpL.at<int>(0, 0) = 4096;
	tmpL.at<int>(0, 1) = 500;
	tmpL.at<int>(0, 2) = 500;
	tmpL.at<int>(0, 3) = 1;
	model->setLayerSizes(tmpL);
	model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM,1.0,1.0);
	model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10000, 1e-4));
	model->setTrainMethod(ANN_MLP::RPROP);

	bool niter = model->train(td);
	cout << "Training done" << endl;

	model->save(argv[5]);

	//static Ptr<TrainData>  td_new = cv::ml::TrainData::create(train_data, ROW_SAMPLE, train_lab);
	//float err = model->calcError(td, true,td->getTestResponses());
	//cv::Mat tst_data=td->getTestSamples();
	//cv::Mat orig_resp = td->getTestResponses;
	
	int sum = 0;
	Mat org = td->getTestResponses();
	Mat results(org.size(),CV_32FC1);
	Mat tmpo = td->getTestSamples();
	/*for (int l = 0; l < tmpo.rows; l++)
		for (int m = 0; m < tmpo.cols; m++)
			cout << tmpo.at<float>(l, m) << endl;*/
	cout << tmpo.rows << " " << tmpo.cols<< endl;
	//Mat idx = td->getTestSampleIdx();
	model->predict(td->getTestSamples(), results);
	
	for (int l = 0; l < results.rows; l++)
		for (int m = 0; m < results.cols; m++)
			cout << results.at<float>(l, m) << " " << org.at<float>(l, m) <<endl;
	cout << results.rows << " " << results.cols << endl;
	/*for (int l = 0; l < org.rows; l++)
	{
		if (org.at<float>(l, 0) == results.at<float>(l, 0))
			sum++;
		cout << org.at<float>(l, 0) << endl;
		cout << results.at<float>(l, 0) << endl;
	}*/
	cout << "Total  is: " << sum << endl;
	int64 t1 = cv::getTickCount();//End

	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "Time elapsed:" << secs / 60 << "Minutes" << endl;
	return 0;
}