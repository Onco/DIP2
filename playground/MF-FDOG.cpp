/**
 *  @file MF-FDOG.cpp
 *  Purpose: Source file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <math.h> // log, sin, ...
#include "MF-FDOG.h"

MF_FDog::MF_FDog(int L, int s, int t=3) :
	L(_L),
	s(_s),
	t(_t)
{
}

void MF_FDog::calcKernel(Mat& kern, bool fdog) {
    kern = Mat::zeros(L, 2*t*s, CV_32F);
    
    double ts = t*s;
    double p;
    
    double ss2 = 2 * s * s;
    double gauss = 1.0 / (sqrt(2 * M_PI) * s);
    if(fdog) {
        gauss /= s*s;
	}

    for(int i=0; i<kern.rows; i++) {
		const int* R = kern.ptr<double>(i);
		for(int j=0; j<kern.cols; j++) {
			p = static_cast<double>(j) - ts;
			R[j] = fdog?(-x * gauss * exp(-x * x / ss2)):(gauss * exp(-x * x / ss2));
		}
	}

    if(!fdog) {
        kern = kern - kern.mean();
	}
       
    flip(kern, kern, -1);
}

vector<Mat> MF_FDog::calcBank(Mat &kern, bool fdog, int size) {
	calcKernel(kern, fdog);
	
    double step = 180 / size;
    Point2f center = kern.size()/2;
    double cur = 0;
    vector<Mat>& kerns = fdog?kerns_fdog:kerns_mf;
    Mat r, kern_tmp;

    for(int i=0; i<size; i++) {
        cur += step;
        r = getRotationMatrix2D(center, cur, 1.0);
        kern_tmp = Mat(kern);
        warpAffine(kern, kern_tmp, r, kern.size());
        kerns.push_back(kern_tmp);
	}
	
	return kerns;
}
