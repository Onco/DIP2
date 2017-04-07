#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;

/// Global variables
Mat src, tmpsrc, erosion_dst, dilation_dst, difference_dst, mask, gauss, roi, blur_dilat, prewitt, prewitt1, prewitt2, kernX, kernY;
double minVal, maxVal;
Point minLoc, maxLoc, mtd1, mtd2, mtd3, avg;
int max_dist, roi_dist;

/** @function main */
int main( int argc, char** argv )
{
  if (argc < 2) return -1;
	
  /// Load an image
  tmpsrc = imread( argv[1] );
  
  if( !tmpsrc.data )
  { return -1; }
  
  src.create(tmpsrc.rows,tmpsrc.cols,CV_8UC1);
  /* int from_to[] = {1,0};
  mixChannels(&tmpsrc, 1, &src, 1, from_to, 1); */
  cvtColor(tmpsrc, src, CV_RGB2GRAY);

  /// Create window
  namedWindow("MinMax Difference", CV_WINDOW_AUTOSIZE);
  
  // calculate max allowed distance (10% of image diagonal)
  max_dist = 0.1*sqrt(src.rows*src.rows+src.cols*src.cols);

  // Min-Max difference value
  int elem_size = ((max_dist/10) > 2) ? (max_dist/10):2;
  Mat element = getStructuringElement(MORPH_RECT, Size(2*elem_size, 2*elem_size), Point(elem_size, elem_size));
  erode(src, erosion_dst, element);
  dilate(src, dilation_dst, element);
  subtract(dilation_dst, erosion_dst, difference_dst);
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
  
  Rect r(avg.x-max_dist, avg.y-max_dist, max_dist*2, max_dist*2);
  roi = Mat(src, r);
  
  imshow("MinMax Difference", roi);
  waitKey(0);
  
  // odstranenie krvenho riecista
	// dilatacia
  roi_dist = 0.1*sqrt(roi.rows*roi.rows+roi.cols*roi.cols);
  elem_size = ((max_dist/10) > 2) ? (max_dist/10):2;
  element = getStructuringElement(MORPH_RECT, Size(2*elem_size, 2*elem_size), Point(elem_size, elem_size));
  dilate(roi, dilation_dst, element);
  imshow("MinMax Difference", dilation_dst);
  waitKey(0);
	// mean filter
  blur(dilation_dst, blur_dilat, Size(3,3));
  
  imshow("MinMax Difference", blur_dilat);
  waitKey(0);
  
  // Prewitt operator
  kernX = (Mat_<int>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
  kernY = (Mat_<int>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
  Mat prewitt1_t, prewitt2_t;
  filter2D(blur_dilat, prewitt1_t, CV_16S, kernX);
  convertScaleAbs(prewitt1_t, prewitt1);
  imshow("MinMax Difference", prewitt1);
  waitKey(0);
  filter2D(blur_dilat, prewitt2_t, CV_16S, kernY);
  convertScaleAbs(prewitt2_t, prewitt2);
  imshow("MinMax Difference", prewitt2);
  waitKey(0);
  addWeighted(prewitt1, 0.5, prewitt2, 0.5, 0, prewitt);
  
  // OSHU 
  threshold(src, mask, 0, 255,THRESH_BINARY+THRESH_OTSU);
  
  // Erosion
  
  
  imshow("MinMax Difference", prewitt);
  waitKey(0);
  
  return 0;
}
