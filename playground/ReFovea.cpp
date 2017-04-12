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

/** Prints out help.
 *  @param p Image to be shown. 
 */
void show(Mat &p) {
  #ifndef VIEW
  resize(p, view, Size(0, 0), 0.1, 0.1);
  imshow("Output", view);
  waitKey(0);
  #endif
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
  
  #ifdef VIEW
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
  cvtColor(input, input, COLOR_BGR2YCrCb);
  
  // set other (then Y) channels to zero
  split(input, channels);
  
  /*proc = Mat(input.rows, input.cols, CV_8UC1);
  int fromTo[] = {0,0};
  mixChannels(&input, 1, &proc, 1, fromTo, 1);*/
  
  cout<<"Grayscale..."<<endl;
  show(channels[0]);
  
  // Preprocessing - move out to special function
  
  // normalize histogram
  equalizeHist(channels[0], channels[0]);
  cout<<"Normalized histogram..."<<endl;
  show(channels[0]);
  
  merge(channels, proc);
  cvtColor(proc, proc, COLOR_YCrCb2BGR);
  cout<<"Enhanced color image..."<<endl;
  show(proc);
  
  // denoise image - use 2 cascade median filters with k=5
  cout<<"Before blur..."<<endl;
  medianBlur(proc, proc, 3);
  medianBlur(proc, proc, 3);
  cout<<"Median filters..."<<endl;
  show(proc);
  
  // enhance contrast - tone curve = LUT?
  /*tonemap = createTonemapDurand(gamma_coeff);
  tonemap->process(proc, proc);
  cout<<"Tonemap..."<<endl;
  show(proc);*/
  
  // Vessel segmentation - using MF-FDOG
  MF_FDoG mfg = MF_FDoG(15, 5);
  mfg.calcBanks();
  
  vector<Mat>& kernels = mfg.getKerns();
  for(vector<Mat>::iterator it = kernels.begin(); it != kernels.end(); ++it) {
	  Mat& x = *it;
	  x *= 255;
	  show(x);
  }
  
  destroyAllWindows();
  
  return 0;
}
