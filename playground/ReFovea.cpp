/**
 *  @file ReFovea.cpp
 *  Purpose: Main source file, contains global defines.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "highgui.h"
//#include <stdlib.h> // malloc/free, atof, abs, exit, rand, srto* + definitions
//#include <stdio.h> // printf, getc, fopen...
#include <math.h> // log, sin, ...
#include <iostream> // cout, cin, ...
#include <string>
#include <unistd.h> // getopt

using namespace cv;
using namespace std;

Mat input; /**< Input image. */

/** Prints out help.
 */
void help(char* name) {
  cout<<"Usage: "<<name<<" image_file"<<endl;
  cout<<"Parameters:\n -h - prints help"<<endl;
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
  
  /// Create window
  namedWindow("Output", CV_WINDOW_AUTOSIZE);
  imshow("Output", channels[0]);
  waitKey(0);
  imshow("Output", channels[1]);
  waitKey(0);
  imshow("Output", channels[2]);
  waitKey(0);
  
  return 0;
}
