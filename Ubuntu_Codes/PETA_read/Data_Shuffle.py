import numpy as np
import random
Inp_file=input("Enter the name of the file with full path containing Image paths:\n")
Out_file=input("Enter the output file name with path and prefix(not the extension):\n")
imgs=[]
with open(Inp_file,'r') as fr:
    for line in fr:
        imgs.append(line.strip())
random.shuffle(imgs)
train=imgs[0:int(0.7*len(imgs))]
val=imgs[int(0.7*len(imgs)):]
with open(Out_file+"_train.txt",'w') as fw1:
    for i in train:
        fw1.write(i+"\n")
with open(Out_file+"_val.txt",'w') as fw2:
    for j in val:
        fw2.write(j+"\n")
fr.close()
fw1.close()
fw2.close()
