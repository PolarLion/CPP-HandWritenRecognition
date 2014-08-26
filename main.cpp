#include "Recognizer.h"


#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
int n = 0;

void *h(void *d)
{
    int a = ++n;
    for ( int i = 0; i < 10; ++i ) {
	printf ("thread %d -------- %d\n", a, i);
	sleep(1);
    }
    return NULL;
}

void keep_window_open()
{
    char c;
    printf("enter a character to exit\n");
    cin >> c;
}


void t(double thrd)
{
    Recognizer t("io/", thrd);
    //t.preprocess("training_set/train-images-idx3-ubyte", "training_set/train-labels-idx1-ubyte");
    //t.train_nn();
    //t.load_nn();
    //t.prepare_top_train("training_set/train-images-idx3-ubyte", "training_set/train-labels-idx1-ubyte");
    //t.train_top_nn();
    t.load_nn_all();
    t.test("training_set/t10k-images-idx3-ubyte", "training_set/t10k-labels-idx1-ubyte");
    //t.test("training_set/train-images-idx3-ubyte", "training_set/train-labels-idx1-ubyte");
}

int tcv2(int argc, const char** argv)
{
    Recognizer t("", 0);
    unsigned char image[784];

    ifstream infile("training_set/train-images-idx3-ubyte");
    if (infile.fail()) {
	printf("error \n");
    }
    //string s;
    //getline(infile, s);
    char bit32[4];
    for (int i = 0; i < 4; ++i ) {
	infile.read(bit32, 4);
    }
    int i = 0;
    while (i < 1) {
	for (int ii = 0; ii < 784; ++ii) {
	    char c[1];
	    infile.read(c, 1);
	    image[ii] = (unsigned char)c[0];
	}
	++i;
    }
    namedWindow("Edge map0", 2);
    Mat img(Size(28, 28), CV_8UC1, image);
    imshow("Edge map0", img);
    infile.close();
    //namedWindow("Edge map", 2);
    int* temp = new int[28*28];
    char *now = new char[28*28];
    t.Kirsch(image, 28, 28, 3, temp);
    for (int i = 0; i < 28 / 4;  ++i) {
	for ( int j = 0; j < 28 / 4; ++j) {
	    double td = t.sixteen_into_one(temp, 28, i * 4, j * 4);
	    //cout << "max " << td << endl;
	    if ( td < Epsilon )
		now[i * 7 + j] = 0;
	    else
		now[i * 7 + j] = td;
	}
    }

    //t.Kirsch(image, 20, 20, 0, image);
    t.canny(image, 28, 28, image);
    //Mat img2(Size(7, 7), CV_8UC1, now);
    Mat img2(Size(28, 28), CV_8UC1, image);
    namedWindow("1", 2);
    imshow("1", img2);
    double m00, m10, m01;
    CvMoments moment;
    //IplImage *src = img.IplImage();
    IplImage src = img2;
    //cvMoments(&src, &moment, 2);
    //m00 = cvGetSpatialMoment(&moment, 0, 0);
    //cout << "m00 " << m00 << endl;
    //CvHuMoments humoment;
    //cvGetHuMoments(&moment, &humoment);
    //cout << "hu :" << humoment.hu1 << endl;
    CvSeq* contour = NULL;

    waitKey();
    //keep_window_open();
    return 0;
}


int main(int argc, const char** argv)
{
    printf ("hello world\n");
    //tcv(argc, argv);
    /* 
       for (double i = 0; i < 1; i += 0.05) {
       t(i);
       }
     */
    t(0);
    //tcv2(argc, argv);

    //cout << t.max(100, 20) << endl;
    //unsigned char bs[9] = { 1, 0, 0,
    //			0, 1, 0,
    //			0, 0, 1};
    //int* f = new int[9];
    //t.Kirsch(bs, 3, 3, 3, f);
    /*
       for (int i = 0; i < 9; ++i) {
       if ( i % 3 == 0 )
       cout << endl;
       cout << f[i] << " ";
       }*/
    //delete f;

    /*pthread_t p[4];
      void *tret[4];
      void *ttt;
      for ( int i = 0; i < 4; ++i ) {
    //n = i;
    pthread_create(&p[i], NULL, h, NULL);
    //pthread_join(p[i], &tret);
    }

    for (int i = 0; i < 4; ++i) {
    pthread_join(p[i], &ttt);
    }

    keep_window_open();*/
    return 0;
}
