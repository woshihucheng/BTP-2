'''
This program trains a HOG+SVM model using n fold validation using the combined dataset
'''
import cv2
import  numpy as np
import svm
import svmutil
import os
import random
from os.path import basename
#np.set_printoptions(threshold=np.nan)
dr="F:\\IET\\BTP\\SVM\\Data\\Cap\\"

#Parameters
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

#Training data
Files=[]
HIST_ALL1=[]
Final_Files=[]
Train_dr=dr
for file in os.listdir(Train_dr+"Positive\\Combined\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Train_dr+"Positive\\Combined\\"+file)
        Final_Files.append(Train_dr+"Positive\\Combined\\"+file)
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL1.append(hist)

y1=np.ones(len(HIST_ALL1))
Files=[]
for file in os.listdir(Train_dr+"Negative\\"):
    if file.endswith(".JPG") or file.endswith(".jpg"):
        Files.append(Train_dr+"Negative\\"+file)
        Final_Files.append(Train_dr+"Negative\\"+file)
HIST_ALL2=[]
for f in Files:
    img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
    hist = hog.compute(img,winStride,padding)
    HIST_ALL2.append(hist)

y2=-1*np.ones(len(HIST_ALL2))
y=np.concatenate([y1,y2])
print "Length of y=" + str(len(y))
x=np.concatenate([HIST_ALL1,HIST_ALL2])[:,:,0]
x=x.tolist()
SEED=25
order=list(range(len(y)))
random.seed(SEED)
random.shuffle(order)
#Train+CV
x_tcv=[x[i] for i in order[0:int(len(order)*3/4)]]
y_tcv=[y[i] for i in order[0:int(len(order)*3/4)]]
#Test Data
x_tst=[x[i] for i in order[int(len(order)*3/4):]]
y_tst=[y[i] for i in order[int(len(order)*3/4):]]
test_fls=[Final_Files[i] for i in order[int(len(order)*3/4):]]
test_writing_dir="F:\\IET\\BTP\\SVM\\Data\\Cap\\New_Folder\\Test\\"
i=0
with open("log_test_images_"+str(SEED)+".txt","w") as fwrite:
    for f in test_fls:
        img=cv2.imread(f,cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
        cv2.imwrite(test_writing_dir+basename(f),img)
        fwrite.write(str(i)+"  "+test_writing_dir+basename(f)+"\n")
        i=i+1
prob=svmutil.svm_problem(y_tcv,x_tcv)
'''C_arr=range(-10,11)
mxACC=0
mxC=0
mxG=0
for i in C_arr:
    for j in C_arr:
        param = svmutil.svm_parameter('-q -t 2 -s 0 -v 10 -c '+ str(2**i) + ' -g ' + str(2**j))
        ACC = svmutil.svm_train(prob, param)
        #print i
        #print j
        if mxACC<ACC:
            mxACC=ACC
            mxC=2**i
            mxG=2**j
print mxC
print mxG
param=svmutil.svm_parameter('-q -t 2 -s 0 -b 1 -c '+ str(mxC) + ' -g ' + str(mxG))'''
param=svmutil.svm_parameter('-t 2 -s 0 -b 1 -c 128 -g 0.03125')
m = svmutil.svm_train(prob, param)
#param = svmutil.svm_parameter('-t 2 -s 2 -c 4 -g 0.001')
#m = svmutil.svm_train(prob, param)

k,b,c=svmutil.svm_predict(y_tst,x_tst,m,'-b 1')
with open("log_test_results_"+str(SEED)+".txt","w") as fwrite2:
    fwrite2.write("SEED Value: "+str(SEED)+"\n")
    fwrite2.write("C Value: "+str(128)+"\n")
    fwrite2.write("gamma Value: "+str(0.03125)+"\n")
    #fwrite2.write("C Value: "+str(mxC)+"\n")
    #fwrite2.write("gamma Value: "+str(mxG)+"\n")
    #fwrite2.write("Training Accuracy: "+str(mxACC)+"\n")
    for i in xrange(len(y_tst)):
        fwrite2.write(str(i)+" Actual Class: " + str(y_tst[i]) + " Predicted class: " + str(k[i])+"\t")
        fwrite2.write(str(y_tst[i]==k[i])+" ")
        fwrite2.write(" Class 1 confidence: " + str(c[i][0]) + " Class -1 confidence : " + str(c[i][1])+"\n")
print k
print b
print c
