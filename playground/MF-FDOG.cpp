/**
 *  @file MF-FDOG.cpp
 *  Purpose: Source file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "MF-FDOG.h"

void showme(Mat& showit) {
	for(int x = 0; x<showit.rows; x++) {
		for(int y = 0; y<showit.cols; y++) {
			if(showit.type() == CV_32F) cout<<setw(5)<<setprecision(2)<<showit.at<float>(x,y)<<" ";
			if(showit.type() == CV_8U) cout<<setw(5)<<setprecision(3)<<showit.at<int>(x,y)<<" ";
		}
		cout<<endl;
	}
}

MF_FDoG::MF_FDoG(float _L, float _s, int _sz, float _t) :
	L(_L),
	s(_s),
	t(_t),
	sz(_sz)
{ 
}

void MF_FDoG::calcKernels() {
	CV_Assert(s!=0);
	CV_Assert(t>0);
	float ts = t*abs(s);
	int ks = static_cast<int>(ceil(sqrt(pow(L,2) + pow(2*ts,2))));
	//cout<<"Size of kernels: "<<ks<<endl;
	if((ks%2) == 0) ks++;
	
	kern_mf = Mat::zeros(ks, ks, CV_32FC1);
	kern_mfm = Mat::zeros(ks, ks, CV_32FC1);
	kern_fdog = Mat::zeros(ks, ks, CV_32FC1);
    
    float ss2 = 2 * s * s;
    float gauss = 1.0 / (sqrt(2 * M_PI) * s);
    float fgauss = gauss / s*s;

	float p;
	int stred = floor(ks/2);
	int sL2 = static_cast<int>(floor(stred - L/2));
	//cout<<"sL2:"<<sL2<<endl;
	int L2s = static_cast<int>(ceil(stred+L/2));
	//cout<<"L2s:"<<L2s<<endl;
	int sts = static_cast<int>(floor(stred - ts));
	//cout<<"sts:"<<sts<<endl;
	int tss = static_cast<int>(ceil(stred + ts));
	//cout<<"tss:"<<tss<<endl;
	
	// flipped i due to filter2D using a correlation instead of convolution (but it is irrelevant as the array is symmetric)
	// similarily for j, but as that is used in Gauss calculation, the flipping is done at assignment
    for(int i=0; i<ks; i++) { 
		for(int j=0; j<ks; j++) {
			// adhere to L/2 boundaries
			//if( (sL2 < i) && (i < L2s) )
			// adhere to +/- sigma*t boundaries
			//if( (sts <= j) && (j <= tss) ) { 
				p = static_cast<float>(j) - stred;
				kern_mf.at<float>(i, j) = gauss * exp(-(p * p) / ss2);
				kern_fdog.at<float>(i, j) = -p * fgauss * exp(-(p * p) / ss2);
			//}
		}
		cout<<endl;
	}
	//kern_mf -= mean(kern_mf);
	cout<<"Sum of matrix: "<<sum(kern_mf)<<endl;
    
    flip(kern_mf, kern_mf, -1);
    flip(kern_fdog, kern_fdog, -1);
    flip(kern_mfm, kern_mfm, -1);
    //kern_mf *= 10;
    //kern_fdog *= 10;
    //normalize(kern_mf, kern_mf, -128, 127, NORM_MINMAX);
    //normalize(kern_fdog, kern_fdog, -128, 127, NORM_MINMAX);
    //showme(kern_mf);
    showme(kern_fdog);
}

void MF_FDoG::calcBanks() {
	calcKernels();
	
    double step = 180 / sz;
    Point2f center = Point2f(kern_mf.size[1]/2, kern_mf.size[0]/2);
    double ang = 0;
    Mat r;
    Mat kt1 = Mat::zeros(kern_mf.size[1], kern_mf.size[0], CV_32FC1); 
    Mat kt2 = Mat::zeros(kern_fdog.size[1], kern_fdog.size[0], CV_32FC1); 
    //cout<<"Size:"<<kern_mf.size[1]<<"x"<<kern_mf.size[0]<<endl;
    //cout<<"Type:"<<kern_mf.type()<<endl;

	kt1 = kern_mf - mean(kern_mf);
	kerns.push_back(make_pair(kt1.clone(), kern_fdog.clone()));
	for(int i=1; i<sz; i++) {
        ang += step;
        r = getRotationMatrix2D(center, ang, 1.0);
        warpAffine(kern_mf, kt1, r, kern_mf.size());
        warpAffine(kern_fdog, kt2, r, kern_fdog.size());
        //showme(kt1);
        kt1 -= mean(kt1);
        //kt2 -= mean(kt2);
        //showme(kt2);
        kerns.push_back(make_pair(kt1.clone(), kt2.clone()));
        //imshow("Output", kt1);
		//waitKey(0);
		//imshow("Output", kt2);
		//waitKey(0);
	}
}
  
