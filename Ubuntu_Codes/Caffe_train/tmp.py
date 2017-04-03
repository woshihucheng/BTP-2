i=0
j=0
sample=0
with open("PETA_val1.txt",'r') as f:
    for line in f:
        i += int(line.strip()[-1])
	j += 1
	if("sampled" in line.strip()):
		sample +=1
		print line.strip()[0:-1]+"2"
	else:
		print line.strip()
#print "Postivees:"+str(i)
#print "Negatives:"+str(j-i)
#print "Sampled: "+ str(sample)
