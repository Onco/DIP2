#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "highgui.h"
//#include <stdlib.h> // malloc/free, atof, abs, exit, rand, srto* + definitions
//#include <stdio.h> // printf, getc, fopen...
#include <math.h> // log, sin, ...
#include <iostream> // cout, cin, ...

using namespace cv;

std::string type2str(int type) {
  std::string r;

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

/// Global variables
Mat src, tmpsrc, erosion_dst, erosion_clr, dilation_dst, difference_dst, mask, gauss, roi, blur_dilat, prewitt, prewitt1, prewitt2, kernX, kernY, bined, circled;
double minVal, maxVal;
 Point minLoc, maxLoc, mtd1, mtd2, mtd3, avg;
int max_dist, roi_dist;
int dp = 4;
int param1 = 100;
int param2 = 420;
std::vector<Vec3f> disks;

/* void computeDilation(int, void*) {
	
} */

void computeHough(int, void*) {
  HoughCircles( erosion_dst, disks, CV_HOUGH_GRADIENT, dp, erosion_dst.rows/8, param1, param2);
	
  cvtColor(erosion_dst, erosion_clr, CV_GRAY2RGB);
  
  if(disks.empty()) {std::cout<<"Empty HOUGH"<<std::endl;}
  else {  
	for(std::vector<Vec3f>::iterator it = disks.begin(); it != disks.end(); ++it)
	{
	  circle(erosion_clr, Point((*it)[0],(*it)[1]), static_cast<int>((*it)[2]), Scalar(255, 0, 0), 2, 8);
	}
  }
  imshow("Hough check", erosion_clr);
  waitKey(0);
}

/** @function main */
int main( int argc, char** argv )
{
  if (argc < 2) return -1;
	
  /// Load an image
  tmpsrc = imread( argv[1] );
  
  if( !tmpsrc.data )
  { return -1; }
  
  src.create(tmpsrc.rows,tmpsrc.cols,CV_8UC1);
  int from_to[] = {1,0};
  mixChannels(&tmpsrc, 1, &src, 1, from_to, 1);

  /// Create window
  namedWindow("MinMax Difference", CV_WINDOW_AUTOSIZE);
  
  // calculate max allowed distance (10% of image diagonal)
  max_dist = 0.1*sqrt(src.rows*src.rows+src.cols*src.cols);

  // Min-Max difference value
  int elem_size = log(max_dist);
  Mat element = getStructuringElement(MORPH_RECT, Size(2*elem_size+1, 2*elem_size+1), Point(elem_size, elem_size));
  std::cout<<"element.type = "<<type2str(element.type())<<std::endl;
  morphologyEx(src, difference_dst, MORPH_GRADIENT, element);
  
  minMaxLoc(difference_dst, &minVal, &maxVal, &minLoc, &maxLoc);
  
  mtd1 = maxLoc;
  
  // Otsu's threshold value
  threshold(src, mask, 0, 255,THRESH_BINARY+THRESH_OTSU);
  minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);
  
  mtd2 = maxLoc;
  
  // Gaussian blur value
  GaussianBlur(src, gauss, Size(3,3), 0);
  minMaxLoc(gauss, &minVal, &maxVal, &minLoc, &maxLoc);
  
  mtd3 = maxLoc;
  
  if(abs(norm(mtd1-mtd2)) > max_dist) {
	  if(abs(norm(mtd2-mtd3)) > max_dist) {
		  if(abs(norm(mtd1-mtd3)) > max_dist) {
			  avg = mtd2;
		  }
		  else {
			  avg = Point((mtd1.x+mtd3.x)/2, (mtd1.y+mtd3.y)/2);
		  }
	  }
	  else {
		  avg = Point((mtd2.x+mtd3.x)/2, (mtd2.y+mtd3.y)/2);
	  }
  }
  else { // mtd1 a mtd2 su pri sebe
	  if(abs(norm(mtd2-mtd3)) > max_dist) {
		  avg = Point((mtd1.x+mtd2.x)/2, (mtd1.y+mtd2.y)/2);
	  }
	  else {
		  avg = Point((mtd1.x+mtd2.x+mtd3.x)/3, (mtd1.y+mtd2.y+mtd3.y)/3);
	  }
  }
  
  circle(tmpsrc, avg, 2, Scalar(255, 0, 0), -1, 8);

  imshow("MinMax Difference", tmpsrc);
  waitKey(0);
  
  int from_to2[] = {2,0};
  mixChannels(&tmpsrc, 1, &src, 1, from_to2, 1);
  
  Rect r(avg.x-max_dist, avg.y-max_dist, max_dist*2, max_dist*2);
  roi = Mat(src, r);
  
  imshow("MinMax Difference", roi);
  waitKey(0);
  
  // odstranenie krvenho riecista
	// dilatacia
  //element = (Mat_<uchar>(5,5) << 0,0,1,0,0, 0,1,1,1,0, 1,1,1,1,1, 0,1,1,1,0, 0,0,1,0,0);
  //element = (Mat_<uchar>(5,5) << 0,0,1,1,1, 0,1,1,1,1, 1,1,1,1,1, 1,1,1,1,0, 1,1,1,0,0);
  element = getStructuringElement(MORPH_RECT, Size(6*log2(roi.rows)+1, 6*log2(roi.cols)+1));
  //std::cout<<"element = "<<std::endl<<element*static_cast<int>(round(mean(roi)[0]))<<std::endl;
  dilate(roi, dilation_dst, element*static_cast<int>(round(mean(roi)[0])), Point(-1,-1), 1);
  imshow("MinMax Difference", dilation_dst);
  waitKey(0);
	// mean filter
  blur(dilation_dst, blur_dilat, Size(2*elem_size+1,2*elem_size+1));
  
  imshow("MinMax Difference", blur_dilat);
  waitKey(0);
  
  // Prewitt operator
  kernX = (Mat_<int>(3,3) << -1, 0, 1, -1,0,1, -1,0,1);
  kernY = (Mat_<int>(3,3) << -1,-1,-1,  0,0,0,  1,1,1);
  Mat prewitt1_t, prewitt2_t;
  filter2D(blur_dilat, prewitt1_t, CV_16S, kernX);
  convertScaleAbs(prewitt1_t, prewitt1);
  filter2D(blur_dilat, prewitt2_t, CV_16S, kernY);
  convertScaleAbs(prewitt2_t, prewitt2);
  addWeighted(prewitt1, 0.5, prewitt2, 0.5, 0, prewitt);
  imshow("MinMax Difference", prewitt);
  waitKey(0);
  
  // OSHU 
  threshold(prewitt, bined, 0, 255,THRESH_BINARY+THRESH_OTSU);
  imshow("MinMax Difference", bined);
  waitKey(0);
  // Erosion 
  element = getStructuringElement(MORPH_RECT, Size(2*log10(bined.rows)+1, 2*log10(bined.cols)+1), Point(-1,-1));
  erode(bined, erosion_dst, element*static_cast<int>(round(mean(bined)[0])));
  imshow("MinMax Difference", erosion_dst);
  waitKey(0);
  
  // Hough Circle Transform
  namedWindow( "Hough check", CV_WINDOW_AUTOSIZE );
  createTrackbar( "DP:\n 0..10", "Hough check", &dp, 10, computeHough );
  createTrackbar( "Element1:\n 0..500", "Hough check", &param1, 500, computeHough );
  createTrackbar( "Element2:\n 0..500", "Hough check", &param2, 500, computeHough );
  computeHough(0, 0);
  
  circled = Mat(tmpsrc,r);
  circle(circled, Point(disks[0][0],disks[0][1]), static_cast<int>(disks[0][2]), Scalar(255, 0, 0), 2, 8);
  
  imshow("MinMax Difference", circled);
  waitKey(0);
  
  return 0;
}
