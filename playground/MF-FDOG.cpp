/**
 *  @file MF-FDOG.cpp
 *  Purpose: Source file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "MF-FDOG.h"

MF_FDoG::MF_FDoG(int _L, int _s, int _sz, int _t) :
	L(_L),
	s(_s),
	t(_t),
	sz(_sz)
{}

void MF_FDoG::calcKernel(bool fdog) {
	CV_Assert(s!=0);
	CV_Assert(t>0);
    Mat &kern = fdog?kern_fdog:kern_mf;
	kern.create(L, 2*t*s, CV_32FC1);

    float ts = t*s;
    
    float ss2 = 2 * s * s;
    float gauss = 1.0 / (sqrt(2 * M_PI) * s);
    if(fdog) {
        gauss /= s*s;
	}

	float p;
    for(int i=0; i<kern.rows; i++) {
		for(int j=0; j<kern.cols; j++) {
			p = static_cast<float>(j) - ts;
			kern.at<float>(i,j) = fdog?(-p * gauss * exp(-p * p / ss2)):(gauss * exp(-p * p / ss2));
		}
	}

    if(!fdog) {
        kern = kern - mean(kern);
	}
    
    flip(kern, kern, -1);
}

void MF_FDoG::calcBank(int size, bool fdog) {
	calcKernel(fdog);
	Mat &kern = fdog?kern_fdog:kern_mf;
	vector<Mat> kerns;
	
    double step = 180 / size;
    Point2f center = Point2f(kern.size[1]/2, kern.size[0]/2);
    double ang = 0;
    Mat r, kern_tmp;

    for(int i=0; i<size; i++) {
        ang += step;
        r = getRotationMatrix2D(center, ang, 1.0);
        warpAffine(kern, kern_tmp, r, kern.size());
        kerns.push_back(kern_tmp);
	}
	 
	if(fdog) kerns_fdog = kerns;
	else kerns_mf = kerns;
}

void MF_FDoG::calcBanks() {
	calcBank(sz, false);
	calcBank(sz, true);
}
