#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include<opencv2/cudalegacy.hpp>
#include<opencv2/cudabgsegm.hpp>
#include<opencv2/cudaobjdetect.hpp>
#include<opencv2/ml.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace cv::ml;
using namespace std;

void load_filenames(vector<string> &filenames,string text_file_dir)
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


int main(int argc,char **argv)
{
	vector<string> pos_filenames;
	vector<string> neg_filenames;
	
	if (argc != 8)
	{
		cout << "Usage: " << argv[0] << " pos_images_text_file neg_images_text_file"
			<< "width height output_file 1/0(BGRA/GRAY) 1/0(Save model/Save weight vector)" << endl;
		cout << "To generate text files use PETA_read.exe file" << endl;
		cout << "Example: PETA_read PETA dataset\\ accessoryHat.txt accessoryHat" << endl;
		return 1;
	}

	int64 t0 = cv::getTickCount();//Beginning of the program

	int code = 0;
	if (atoi(argv[6]) == 1)
		code = CV_BGR2BGRA;
	else
		code = CV_BGR2GRAY;

		
	string pos_dir = argv[1];//Directory for positive images
	string neg_dir = argv[2];//Directory for positive images
	cv::Size sz(atoi(argv[3]), atoi(argv[4])); //Opencv supports only multiple of 8x8 winStride as of now
	
	
	
	/*Setting paramters for HOG*/
	cv::Size  winSize = sz;
	cv::Size  blockSize(16, 16);//Supports only 16x16
	cv::Size  blockStride(8, 8);//Only multiple of cellsize
	cv::Size  cellSize(8, 8); //Supports only 8x8 
	int nbins = 9;
	cv::Size winStride(8, 8);
	cv::Size padding(0, 0);

	/*Setting params for HOG->compute*/
	int feat_sz = ((int)sz.height/cellSize.height - 1) * ((int)sz.width / cellSize.width - 1) * 36;
	cout << "Feature size is: " << feat_sz << endl;
	cv::Ptr<cv::cuda::HOG> d_hog = cv::cuda::HOG::create(winSize, blockSize, blockStride, cellSize, nbins);
	
	load_filenames(pos_filenames, pos_dir);
	
	cv::Mat train_data(0,feat_sz,CV_32FC1);//HOG freatures Matrix where each row correspondes to a sample
	for (size_t i = 0; i < pos_filenames.size(); ++i)
	{
		Mat src = imread(pos_filenames[i]);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;
		resize(src, src, sz, 0, 0, INTER_AREA);
		cuda::GpuMat d_src;
		d_src.upload(src);
		cv::cuda::cvtColor(d_src, d_src,code);
		d_hog->compute(d_src, d_src);
		d_src.reshape(0,1);
		cv::Mat h_src(d_src);
		train_data.push_back(h_src.clone());
	}
	cout << "Positive image read" << endl;
	cv::Mat pos_lab(train_data.rows,1, CV_32SC1,Scalar(1));//Matrix of 1s
	//pos_lab = Scalar(1);

	load_filenames(neg_filenames, neg_dir);
	long int k = 0;
	size_t limit = neg_filenames.size();
	if (pos_filenames.size() < neg_filenames.size())
		limit = (long int) pos_filenames.size();
	for (size_t i = 0; i < limit; ++i)
	{
		srand((unsigned int)time(NULL));
		Mat src = imread(neg_filenames[rand() % neg_filenames.size()]);

		/*if (!src.data)
			cerr << "Problem loading image!!!" << endl;
		resize(src, src, sz, 0, 0, INTER_AREA);
		cuda::GpuMat d_src;
		d_src.upload(src);
		cv::cuda::cvtColor(d_src, d_src, code);
		d_hog->compute(d_src, d_src);
		d_src.reshape(0, 1);
		cv::Mat h_src(d_src);
		train_data.push_back(h_src.clone());*/

		/*Sampling Each Negative*/
		srand((unsigned int)time(NULL));//Setting the seed


		Rect samp_ROI;
		samp_ROI.height = rand() % (sz.height / 2) + 1;
		samp_ROI.width = rand() % (sz.width / 2) + 1;
		samp_ROI.x = rand() % (sz.width - samp_ROI.width);
		samp_ROI.y = rand() % (sz.height - samp_ROI.height);
		cv::Mat h_src1(src, samp_ROI);
		resize(h_src1, h_src1, sz, 0, 0, INTER_AREA);
		cuda::GpuMat d_src1;
		d_src1.upload(h_src1);
		cv::cuda::cvtColor(d_src1, d_src1, code);
		d_hog->compute(d_src1, d_src1);
		d_src1.reshape(0, 1);
		cv::Mat h_new(d_src1);
		train_data.push_back(h_new.clone());
		k+=1;
	}

	cout << "Reading files done" << endl;
	cout << "Number of positive sameples=" << pos_filenames.size()<<endl;
	cout << "Number of negative sameples=" << k << endl;
	cv::Mat neg_lab(k, 1, CV_32SC1, Scalar(-1));//Matrix of -1s
	cv::Mat train_lab;
	cv::vconcat(pos_lab, neg_lab, train_lab);
	cout << "Generating Labels done" << endl;
	/*Training*/
	Ptr<SVM> svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	//svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100000, 1e-6));
	svm->setGamma(1);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1);
	svm->setC(0.1); // Initial C
	svm->setType(SVM::EPS_SVR);
	cout << "Setting Params done" << endl;
	svm->train(train_data, ROW_SAMPLE, train_lab);
	static Ptr<TrainData>  td = cv::ml::TrainData::create(train_data,ROW_SAMPLE,train_lab);
	cout << "Generating train data pointer done" << endl;
	//svm->trainAuto(td, 10, ParamGrid(1e-2,1e2, 10), ParamGrid(1e-2, 1e2, 10), SVM::getDefaultGrid(SVM::P),
	//SVM::getDefaultGrid(SVM::NU), SVM::getDefaultGrid(SVM::COEF), SVM::getDefaultGrid(SVM::DEGREE),true);
	cout << "Training done" << endl;
	//svm->train(td);
	cout << svm->getC() << endl;
	if (atoi(argv[7]) == 1)
	{
		svm->save(argv[5]);
		return 0;
	}
	/*Converting weights into float vector*/
	
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	vector<float> hog_detector;
	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;


	/*Save SVM model to use for opencv function for HOG*/
	ofstream myfile;
	myfile.open(argv[5]);
	for (int i = 0; i < hog_detector.size()-1; i++)
	{
		myfile << hog_detector[i] << endl;
	}
	myfile << hog_detector[hog_detector.size() - 1];
	myfile.close();

	int64 t1 = cv::getTickCount();//End
	double secs = (t1 - t0) / cv::getTickFrequency();
	cout << "Time elapsed:" << secs/60 << "Minutes" << endl;
	return 0;
}