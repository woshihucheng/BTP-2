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
#include <fstream>
#include <iostream>
using namespace cv;
using namespace cv::ml;
using namespace std;

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void get_svm_detector_OCV(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void get_svm_detector(string &file_path, vector< float > & hog_detector, const Size& winSize, const Size& cellSize)
{
	long int feat_sz = ((long int)winSize.height / cellSize.height - 1) * ((long int)winSize.width / cellSize.width - 1) * 36 + 1;
	hog_detector.reserve(feat_sz);
	ifstream svm_stream(file_path);
	string line;
	if (svm_stream.is_open())
	{
		while (getline(svm_stream, line))
		{
			hog_detector.push_back(atof(line.c_str()));
			cout << atof(line.c_str()) << endl;
		}
	}
	else
	{
		cout << "Could not open file" << endl;
	}
}

int main(int argc, char** argv)
{
	if (argc != 10)
	{
		cout << " Usage: Opencv_try Video_Directory minArea randseed svm_file win_width win_height numlevs"
			 << "1/0(BGRA/GRAY) 1/0(Save model/Save weight vector)" << endl;
		return -1;
	}

	int code = 0;
	if (atoi(argv[8]) == 1)
		code = CV_BGR2BGRA;
	else
		code = CV_BGR2GRAY;
	Size win_stride(8, 8);
	Size win_size(atoi(argv[5]), atoi(argv[6]));
	Size block_size(16, 16);
	Size block_stride(8, 8);
	Size cell_size(8, 8);
	int nbins = 9;
	vector< float > hog_detector;
	if (atoi(argv[9]) == 1)
	{
		Ptr<SVM> svm = StatModel::load<SVM>(argv[4]);
		get_svm_detector_OCV(svm, hog_detector);
	}
	else
	{
		string svm_file = argv[4];
		get_svm_detector(svm_file, hog_detector, win_size, cell_size);
	}
	VideoCapture cap(argv[1]); // open the video file

	if (!cap.isOpened())  // check whether video capture object is opened or not
		return -1;
	Mat h_frame, h_foreground, h_fgmask;//Host frames
	cuda::GpuMat d_foreground, d_fgmask; //Device(Gpu) images
	
	/*Imshow windows initialization*/
	namedWindow("Foreground", 1);
	namedWindow("h_ROI", 1);
	namedWindow("a", 1);
	
	Ptr<BackgroundSubtractor> fgd = cuda::createBackgroundSubtractorFGD();
	Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
	Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractor> gmg = cuda::createBackgroundSubtractorGMG(40);
	vector<Ptr<BackgroundSubtractor>> v_models;
	v_models.push_back(fgd);
	v_models.push_back(mog);
	v_models.push_back(mog2);
	v_models.push_back(gmg);
	/*Initial run for FGD*/
	//cout << "abc";
	cap >> h_frame;
	cuda::GpuMat d_frame(h_frame);
	gmg->apply(d_frame, d_fgmask);

	//Params for bounding box
	RNG rng(atoi(argv[3]));
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	int mod = 3;
	long long int frm_cnt = 0;

	//Params for HOG
	

	cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins);
	//Mat detector = gpu_hog->getDefaultPeopleDetector();
	//gpu_hog->setSVMDetector(detector);
	gpu_hog->setNumLevels(atoi(argv[7]));
	gpu_hog->setHitThreshold(1.5);
	gpu_hog->setWinStride(win_stride);
	gpu_hog->setScaleFactor(1.05);
	gpu_hog->setGroupThreshold(8);
	gpu_hog->setSVMDetector(hog_detector);
	vector<Rect> found;
	cv::Mat h_ROI;

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
		if(mod==0)
			v_models[mod]->apply(d_frame, d_fgmask,0.01);
		else
			v_models[mod]->apply(d_frame, d_fgmask);//Background subtraction
		d_foreground.create(d_frame.size(), d_frame.type());
		d_foreground.setTo(Scalar::all(0));
		d_frame.copyTo(d_foreground, d_fgmask);
		d_foreground.download(h_foreground);
		d_fgmask.download(h_fgmask);
		//cout << "bad";
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
				//if (boundRect[i].width > 64 && boundRect[i].height > 128)
				//{
					Mat ROI(h_frame, boundRect[i]);
					cout << boundRect[i].height << " " << boundRect[i].width << endl;
					resize(ROI,ROI,cv::Size(40,96),0,0,INTER_LINEAR);
					cv::cuda::GpuMat d_ROI(ROI);
					//d_ROI.upload();
					//cout << d_ROI.empty();
					//cout << d_ROI.col;
					cv::cuda::cvtColor(d_ROI, d_ROI, code);
					d_ROI.download(h_ROI);
					gpu_hog->detectMultiScale(d_ROI, found);
					cout << "a" << endl ;
					imshow("a", h_ROI);
					for (size_t j = 0; j < found.size(); j++)
					{
						cout << "Found";
						Rect r = found[j];
						//rectangle(h_frame, r.tl() + boundRect[i].tl(), r.br() + boundRect[i].tl(), Scalar(0, 255, 0), 3);
						cv::Mat tmp_ROI(h_ROI, r);
						
						imshow("h_ROI", tmp_ROI);
						
					}
				//}
				
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