CC=g++

CXXFLAGS = -O2 -Wall

INCLUDE = -I/usr/local/cuda/include \
			   -I/usr/include/google/ \
                           -I/media/ghost/00521D65521D60A8/IET/Softwares/Libs/Ubuntu/caffe-master/build/install/include/ \
			   $(shell pkg-config --cflags opencv) \
			   -I/home/ghost/anaconda2/include/python2.7 \

LIBRARY = -L/usr/lib/x86_64-linux-gnu/ -lprotobuf \
					-L/usr/local/lib/ -lglog \
					-L/usr/local/cuda/lib64/ -lcudart -lcublas -lcurand \
					-L/usr/lib/x86_64-linux-gnu/ -lm -lpthread -lavutil -lavformat -lavcodec -lswscale -lboost_system\
					$(shell pkg-config --libs opencv) \
                                        -L/usr/local/lib/ -lopencv_imgproc\
					-L/home/ghost/anaconda2/lib/ -lpython2.7 \
					-L/usr/lib/x86_64-linux-gnu/ -lrt \
					-L/media/ghost/00521D65521D60A8/IET/Softwares/Libs/Ubuntu/caffe-master/build/install/lib/ -lcaffe

all:
	$(CC) $(INCLUDE) $(OBJS) Caffe_Helper.cpp -c $(LIBRARY)
	$(CC) $(INCLUDE) $(OBJS) Caffe_Helper.o BG_Caffe.cpp -o BG_Caffe $(LIBRARY)
