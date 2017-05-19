/**
 *  @file ReFovea.cpp
 *  Purpose: Main source file, contains global defines.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
//#include "highgui.h"
//#include <stdlib.h> // malloc/free, atof, abs, exit, rand, srto* + definitions
//#include <stdio.h> // printf, getc, fopen...
#include <math.h> // log, sin, ...
#include <iostream> // cout, cin, ...
#include <unistd.h> // getopt
#include <string>
#include <vector>

#include "MF-FDOG.h"

using namespace cv;
using namespace std;

Mat input; /**< Input image. */
Mat view; /**< Matrix used for viewing. */
Mat proc; /**< Processed image. */
Mat im;
Ptr<Tonemap> tonemap; /**< Tonemap for contrast enhancement. */
float gamma_coeff = 2.2f; /**< Gamma coefficient to be used for tonemap. */

/** Prints out help.
 */
void help(char* name) {
  cout<<"Usage: "<<name<<" image_file"<<endl;
  cout<<"Parameters:\n -h - prints help"<<endl;
}

/** Shows scaled image.
 *  @param p Image to be shown. 
 */
void show(Mat &p) {
  #ifndef VIEW
  resize(p, view, Size(0, 0), 0.2, 0.2);
  imshow("Output", view);
  waitKey(0);
  #endif
}

/** Prints out image.
 *  @param showit Image to be printed out. 
 */
void showm(Mat& showit) {
	for(int x = 0; x<showit.rows; x++) {
		for(int y = 0; y<showit.cols; y++) {
			if(showit.type() == CV_32F) cout<<setw(10)<<setprecision(7)<<showit.at<float>(x,y)<<" ";
			if(showit.type() == CV_8U) cout<<setw(10)<<setprecision(7)<<showit.at<int>(x,y)<<" ";
		}
		cout<<endl;
	}
}

/** Converts type of Mat object to a string.
 *  @param type Integer representing type of Mat object.
 *  @return Returns string containing type of Mat object.
 */
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

/** Main function.
 * @param argc Number of input arguments.
 * @param argv Pointer to the arguments.
 * @return Returns 0 on success, error code number otherwise.
 */
int main( int argc, char** argv )
{
  int fl, opt;
  string img;
  
  while ((opt = getopt(argc, argv, "h")) != -1) {
    switch (opt) {
      case 'h':
        help(argv[0]);
        return 0;
        break;
      default:
        cerr<<"Incorrect parameters!"<<endl;
        help(argv[0]);
        return -1;
	  }
  }
	
  if((argc <= 1) || (optind == argc)) {
    cerr<<"No input images specified!"<<endl;
    return -1;
  }
  
  // Load an image
  input = imread( argv[optind] );
  cout<<"Type: "<<type2str(input.type())<<endl;
  
  if( input.empty() ) { 
    cerr<<"Input image is empty!"<<endl;
    return -1;
  }
  
  if(++optind < argc) {
    cerr<<"Too many arguments!"<<endl;
    return -1;
  }
  
  // Show each channel
  vector<Mat> channels;
  split(input, channels);
  
  #ifndef VIEW
  /// Create window
  namedWindow("Output", CV_WINDOW_AUTOSIZE);
  for(int i=0; i<3; i++) {
    switch(i) {
      case 0: cout<<"Red"<<endl; break;
      case 1: cout<<"Green"<<endl; break;
      case 2: cout<<"Blue"<<endl; break;
      default: cout<<"Error"<<endl;
    }
    show(channels[i]);
  }
  #endif

  // convert to YCrCb
  //cvtColor(input, input, COLOR_BGR2YCrCb);
  
  // set other (then Y) channels to zero
  //split(input, channels);
  
  cout<<"Grayscale..."<<endl;
  //show(channels[0]);
  
  proc = Mat(input.rows, input.cols, CV_8UC1);
  int fromTo[] = {1,0};
  mixChannels(&input, 1, &proc, 1, fromTo, 1);
  show(proc);
  
  // Preprocessing - move out to special function
  
  // normalize histogram
  //equalizeHist(channels[0], channels[0]);
  equalizeHist(proc, proc);
  cout<<"Normalized histogram..."<<endl;
  //show(channels[0]);
  show(proc);
  
  //merge(channels, input);
  //cvtColor(input, input, COLOR_YCrCb2BGR);
  //cout<<"Enhanced color image..."<<endl;

// GRAYING !!!
  //cvtColor(proc, proc, COLOR_BGR2GRAY);
  //show(proc);
  
  // denoise image - use 2 cascade median filters with k=3
  cout<<"Before blur..."<<endl;
  medianBlur(proc, proc, 3);
  medianBlur(proc, proc, 3);
  cout<<"Median filters..."<<endl;
  show(proc);
  
  cout<<"Input Mat type: "<<type2str(proc.type());
  bitwise_not(proc,proc);
  show(proc);
  
  // enhance contrast - tone curve = LUT?
  /*tonemap = createTonemapDurand(gamma_coeff);
  tonemap->process(proc, proc);
  cout<<"Tonemap..."<<endl;
  show(proc);*/
  
  // Vessel segmentation - using MF-FDOG
  MF_FDoG mfg = MF_FDoG(9, 1.5, 12);
  MF_FDoG mfg2 = MF_FDoG(5, 1, 12);
  mfg.calcBanks();
  mfg2.calcBanks();
  
  vector< pair<Mat, Mat> >& kernels = mfg.getKerns();
  vector< pair<Mat, Mat> >& kernels2 = mfg2.getKerns();
  vector< pair<Mat, Mat> > responses;
  Mat mf_proc, fdog_proc, proc2, mf_proc2, fdog_proc2, t, t2;
  Rect ROI(50,50, 20, 20);

  int c=0;
  cout<<"Calculating response: "<<flush;
  for(int i=0; i<kernels.size(); i++) { //vector< pair<Mat, Mat> >::iterator it = kernels.begin(); it != kernels.end(); ++it) {
    //k.convertTo(k, CV_8UC1);
    show(proc);
	  filter2D(proc, mf_proc, -1, kernels[i].first);
    //show(mf_proc);
    filter2D(proc, fdog_proc, -1, kernels[i].second);
    show(fdog_proc);
    blur(fdog_proc, fdog_proc, Size(31, 31));
    //normalize(fdog_proc, fdog_proc, 0, 1, NORM_MINMAX);
    t = 1+fdog_proc;
    //show(t);
    cout<<"Mean of mf_proc: "<<mean(mf_proc)[0]<<endl;
    cout<<"Mean of fdog_proc: "<<mean(t)[0]<<endl;
    //fdog_proc = t.mul(2.3 * mean(mf_proc)[0]);
    //Mat shw = fdog_proc(ROI);
    //showm(shw);
    //compare(mf_proc, fdog_proc, mf_proc, CMP_GE);
    
    
    filter2D(proc, mf_proc2, -1, kernels2[i].first);
    filter2D(proc, fdog_proc2, -1, kernels2[i].second);
    //blur(fdog_proc2, fdog_proc2, Size(31, 31));
    //normalize(fdog_proc2, fdog_proc2, 0, 1, NORM_MINMAX);
    t2 = 1+fdog_proc2;
    //fdog_proc2 = t2.mul(2.3 * mean(mf_proc2)[0]);
    //compare(mf_proc2, fdog_proc2, mf_proc2, CMP_GE);

    //imshow("Output",(*it).first);
    //waitKey(0);
    //imshow("Output",(*it).second);
    //waitKey(0);
    //filter2D(proc, fdog_proc, -1, *it);
    //show(mf_proc);
	  responses.push_back( make_pair(mf_proc.clone(), fdog_proc.clone()) );
    cout<<" "<<c++<<flush;
  }
  cout<<"."<<endl;
  
  Mat outx = Mat::zeros(responses[0].first.size(), responses[0].first.type());
  Mat outy = Mat::zeros(responses[0].second.size(), responses[0].second.type());
	for(vector< pair<Mat, Mat> >::iterator it = responses.begin(); it != responses.end(); ++it){
    //show((*it).first);
    //show((*it).second);
    max((*it).first, outx, outx);
    max((*it).second, outy, outy);
	}
  show(outx);
  show(outy);
  
  destroyAllWindows();
  
  return 0;
}
   
