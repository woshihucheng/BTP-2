import cv2
import  numpy as np
import svm
import svmutil
import os
#np.set_printoptions(threshold=np.nan)
dr="F:\\IET\\BTP\\SVM\\Data\\Cap\\"

#print img
#cv2.imshow("resized", img)
#cv2.waitKey(0)
#hog= cv2.HOGDescriptor()
winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 0
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold)
winStride = (8,8)
padding = (0,0)
#locations = ((10,20),)
Files=[]
HIST_ALL=[]
Train_dr=dr
for file in os.listdir(Train_dr+"Positive\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Train_dr+"Positive\\"+file)
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL.append(hist)
HIST_ALL1=np.array(HIST_ALL)
print HIST_ALL1.shape
y1=np.ones((HIST_ALL1.shape[0],1))
Files=[]
for file in os.listdir(Train_dr+"Negative\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Train_dr+"Negative\\"+file)
HIST_ALL=[]
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL.append(hist)
HIST_ALL2=np.array(HIST_ALL)
print HIST_ALL2.shape
y2=-1*np.ones((HIST_ALL2.shape[0],1))
y=np.concatenate([y1,y2])
x=np.concatenate([HIST_ALL1,HIST_ALL2])[:,:,0]
#print x.shape
print len(y[:])
x=x.tolist()
#print (y)
prob=svmutil.svm_problem(y[:],x[:][:])
param = svmutil.svm_parameter('-t 2 -s 2 -c 4 -g 0.001')
m = svmutil.svm_train(prob, param)

Test_dr=dr+"Test\\"
Files = []
HIST_ALL=[]
for file in os.listdir(Test_dr+"Positive\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Test_dr+"Positive\\"+file)
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL.append(hist)
HIST_ALL1=np.array(HIST_ALL)
y1=np.ones((HIST_ALL1.shape[0],1))
Files=[]
for file in os.listdir(Test_dr+"Negative\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Test_dr+"Negative\\"+file)
HIST_ALL=[]
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL.append(hist)
HIST_ALL2=np.array(HIST_ALL)
y2=-1*np.ones((HIST_ALL2.shape[0],1))
y=np.concatenate([y1,y2])
x=np.concatenate([HIST_ALL1,HIST_ALL2])[:,:,0]
x=x.tolist()


k,b,c=svmutil.svm_predict(y[:],x[:][:],m)
print k,b,c
