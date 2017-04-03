import collections
path_to_file=input("Enter the file prefix with full path that was used for splitting training and validation files")
lst=[]
with open(path_to_file+"_train.txt",'r') as f:
	for line in f:
		lst.append(line.strip()[-1])
counter=collections.Counter(lst)
print "Following are classes with number of samples in training file:\n"
print counter

lst=[]
with open(path_to_file+"_val.txt",'r') as f:
	for line in f:
		lst.append(line.strip()[-1])
counter=collections.Counter(lst)
print "Following are classes with number of samples in validation file:\n"
print counter
