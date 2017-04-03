'''
Change
- Apply SVM for each window rather than collecting all the histogram for a window
- Use existing models
'''
import cv2
import  numpy as np
import svm
import svmutil
import os
import sys
print sys.executable

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

dr_mod = input("Enter the directory for precomputed model")
m = svmutil.svm_load_model(dr_mod)
cap = cv2.VideoCapture("F:\\IET\\BTP\\SVM\\Data\\Robbery.mp4")
print "a"
frame = 0
while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(300,200))
    #print ret
    if not (ret):
        break
    for i in xrange(0,frame.shape[1]-64,winStride[0]):
        for j in xrange(0, frame.shape[0]-128, winStride[1]):
            #print frame.shape
            #print frame[j:j+128,i:i+64.shape
            hist_f = hog.compute(frame[j:j+128,i:i+64,:],winStride,padding)
            tmpt = [hist_f[0:350,0].tolist(), np.zeros(350).tolist()]
            #print len(hist_f[:,0].tolist())
            print [0]*len(tmpt)
            k,b,c=svmutil.svm_predict([0]*len(tmpt),tmpt,m,options='-q -b 1')
            print k
            if k==1:
                print i,j
                break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#k,b,c=svmutil.svm_predict(y[:],x[:][:],m,'-b 1')
#print k,b,c
