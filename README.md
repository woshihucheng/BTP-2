### B.Tech-Project - Person Identification based on material attributes

./Before_Midterm : Code submitted before mid-term evaluation
	Train_CV python scripts : Train a HOG+SVM classifier for cap images
	Video_Test python scripts : Test the trained classifier for caps on videos
./Caffe_Train :  Training files for detecting a person with a backpack in caffe for windows (simple ANN with 2 hidden layers woth 500 hidden nodes each)
./Person_Det_BGSUB : Visual Studio projects for the BTP.
./Ubuntu_Codes : Codes that run on Ubuntu(Caffe,OpenCV 3.2.0)

Ubuntu_Codes/PETA_read : The scripts in this directory are for augmenting PETA dataset(Randomly Cropping, Multiplication, Addition, Histogram Equalizaton, and Affien Transofrmation based scaling),creating a file which contains full path for all images and classes of each file, Shuffling this file for training and validation files, and for counting number of samples for each classes.

Ubuntu_Codes/Caffe_Train : Files for training caffe models.

Ubuntu_Codes/Caffe_OCV : Main file which detects a backpack from a video using provided caffe trained model.
