import os
import glob
import ntpath
import cv2
import random
import numpy as np
PETA_dir = input("Input full path of the main directory of PETA dataset \
which contains all the foldes of datatbases(3DPeS CUHK):\n")
PETA_dir += "/"
Out_dir = input("Enter the path of an EMPTY folder where the images will be stored. \
Images will be stored as datasetname_Img_filename in given Out_dir:\n")
os.mkdir(Out_dir)
Out_file = input("Input full path of the output file where you \
want to save the file containing paths and labels:\n")
Attribute = input("Enter the name of the attribute for which you want to \
generate the data. See names of all the attributes in \
main folder of PETA: file  README:\n")
RGB_GRAY = input("Images should be saved in RGB or GRAY?[R/G]:\n")
win_height = int(input("Enter the height of new images:\n"))
win_width = int(input("Enter the width of new images:\n"))
class_no = input("Enter the class number which should not be 0(0 is for background or random cropped images):\n")
RG = 0#flag 1 for RGB and 0 for Gray
if RGB_GRAY=="G":
    RG=0
elif RGB_GRAY == "R":
    RG=1
else:
    quit()

print("Each of the following options will create a new image\n")
Aff = input("Do you want to zoom(Affine Transformation) image?[Y/N]:\n")
flag_Aff=0
if Aff=="Y":
    flag_Aff=1
else:
    flag_Aff=0

Mul = input("Do you want to multiply image(multiplied with 0.5 and 1.5) image?[Y/N]:\n")
flag_Mul=0
if Mul=="Y":
    flag_Mul=1
else:
    flag_Mul=0

Add = input("Do you want to add a value to image(added -50 and 50) image?[Y/N]:\n")
flag_Add=0
if Add=="Y":
    flag_Add=1
else:
    flag_Add=0

Hist = input("Do you want to add an image with histogram equalized?[Y/N]:\n")
flag_Hist=0
if Hist=="Y":
    flag_Hist=1
else:
    flag_Hist=0

Crop = input("Do you want to do random sampling(Used only for bacground class(Class_num=0))?[Y/N]:\n")
flag_Crop=0
if Crop=="Y":
    flag_Crop=1
else:
    flag_Crop=0    

datasets=os.listdir(PETA_dir)
datasets1=[]
for ds in datasets:
        ans=input("Select database "+ds+"? [Y/N]:")#Some databases have very low resolution
        if ans=="Y" :
            datasets1.append(ds)
under="_"
with open(Out_file,'a') as f:
    for ds in datasets1:
        #Label.txt in each dataset folder conatain information for each image or a subject
        with open(PETA_dir+ds+"/archive/Label.txt",'r') as flbl:
            for line in flbl:
                ln=line.strip()#Removing newline from each line
                if Attribute in ln:
                    #CUHK database's Label.txt contains the image name and not the prefix of the subject
                    #All other dataabses contains a subjectname_* files for asubject
                    if not ds == "CUHK":
                        under="_"
                    else:
                        under=""
                    fl_pre = ln.split()[0]#Name of the subject or for CUHK the name of the image file
                    #Searching for images with given prefix in the directory
                    for files in glob.glob(PETA_dir+ds+"/archive/"+fl_pre+under+"*"):
                        img=cv2.imread(files,RG)#Loading image in RGB or gray straight
                        head,tail=ntpath.split(files)#Returns directory as head and filename as tail
                        img=cv2.resize(img,(win_width,win_height))                            
                        cv2.imwrite(Out_dir+"/"+ds+"_Img_"+tail,img)
                        f.write(Out_dir+"/"+ds+"_Img_"+tail+" "+class_no+"\n")
                        if flag_Aff==1:
                            pts1 = np.float32([[10,10],[54,10],[54,54],[10,54]])
                            pts2 = np.float32([[0,0],[64,0],[64,64],[0,64]])
                            M = cv2.getPerspectiveTransform(pts1,pts2)
                            dst = cv2.warpPerspective(img,M,(win_width,win_height))
                            str_pref="Img_Scaled_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Aff_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                        if flag_Crop==1:
                            #Randomly Selecting and y points
                            x = random.randrange(0,img.shape[1]-10)
                            y = random.randrange(0,img.shape[0]-10)
                            dst=img[y:y+10,x:x+10]
                            dst=cv2.resize(dst,(win_width,win_height))
                            str_pref="Img_Cropped_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                        if flag_Mul==1:
                            tmp = 1.5 * np.ones(img.shape,dtype=np.uint8)
                            dst=0
                            if RG==1:
                                dst = cv2.multiply(tmp,img,dtype=cv2.CV_8UC3)
                            else:
                                dst = cv2.multiply(tmp,img,dtype=cv2.CV_8UC1)
                            str_pref="Img_Multiplied1_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Mul1_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                            tmp = 0.5 * np.ones(img.shape,dtype=np.uint8)
                            dst=0
                            if RG==1:
                                dst = cv2.multiply(tmp,img,dtype=cv2.CV_8UC3)
                            else:
                                dst = cv2.multiply(tmp,img,dtype=cv2.CV_8UC1)
                            str_pref="Img_Multiplied2_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Mul2_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                        if flag_Add==1:
                            tmp = 50 * np.ones(img.shape,dtype=np.uint8)
                            dst=0
                            if RG==1:
                                dst = cv2.add(tmp,img,dtype=cv2.CV_8UC3)
                            else:
                                dst = cv2.add(tmp,img,dtype=cv2.CV_8UC1)
                            str_pref="Img_Added1_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Add1_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                            tmp = -50 * np.ones(img.shape,dtype=np.uint8)
                            dst=0
                            if RG==1:
                                dst = cv2.add(tmp,img,dtype=cv2.CV_8UC3)
                            else:
                                dst = cv2.add(tmp,img,dtype=cv2.CV_8UC1)
                            str_pref="Img_Added2_"
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Mul2_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
                        if flag_Hist==1:
                            dst=0
                            if RG==1:
                                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                                dst = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                            else:
                                dst=cv2.equalizeHist(img)
                            str_pref="Img_Equalized_"
                            dst=cv2.resize(dst,(win_width,win_height))
                            cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst)
                            f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" "+class_no+"\n")
                            if flag_Crop==1:
                                #Randomly Selecting and y points
                                x = random.randrange(0,dst.shape[1]-10)
                                y = random.randrange(0,dst.shape[0]-10)
                                dst2=dst[y:y+10,x:x+10]
                                dst2=cv2.resize(dst2,(win_width,win_height))
                                str_pref="Img_Hist_Cropped_"
                                cv2.imwrite(Out_dir+"/"+ds+"_"+str_pref+tail,dst2)
                                f.write(Out_dir+"/"+ds+"_"+str_pref+tail+" 0"+"\n")
f.close()
flbl.close()
