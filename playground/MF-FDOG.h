/**
 *  @file MF-FDOG.h
 *  Purpose: Header file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#ifndef MF_FDOG_H
#define MF_FDOG_H 1

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include <math.h> // log, sin, ...
#include <vector>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

class MF_FDoG {
  public:
    /** Constructor of MF_FDoG
     *  @param L Length of the neighborhood along the y-axis to smooth noise.
     *  @param s Variance of Gaussian kernels in banks.
     *  @param sz Number of kernels to be created. Kernels are obtained by rotating the base kernel.
     *  @param t Constant selecting whole Gaussian in kernels.
     */
     MF_FDoG(float _L, float _s, int _sz, float _t=3);
  
    /** Method applies kernel bank to input image and provides maximal response as result.
     *  @param src Reference to source image.
     *  @param dst Reference to output image.
     *  @param kerns Kernel bank to be applied.
     */
    void process(Mat &src, Mat &dst, vector<Mat> &kerns);


    /** Method calculating kernel banks for MF and FDoG.
     */    
    void calcBanks();
    
    /** Method calculating base kernel for MF and FDoG.
     */
    void calcKernels();

    /** Accessor to calculated kernel banks for MF and FDoG.
     */
    vector< pair<Mat, Mat> >& getKerns();
    
    /** Return kernel size.
     */
    Size getKernSize();

  //private:
    float L;    /**< Length of the neighborhood along the y-axis to smooth noise. */
    float s;  /**< Variance of the Gaussian. */
    float t;  /**< Constant, sigma scale to get to Gaussian width. */
    Mat kern_mf;  /**< Base kernel for Gaussian Matched Filter. */
    Mat kern_mfm;
    Mat kern_fdog; /**< Base kernel for FDoG Filter. */
    vector< pair<Mat, Mat> > kerns; /**< Kernel banks for Gaussian Matched Filter and FDoG Filter. */
    int sz;  /**< Size of kernel banks. */
};

inline vector< pair<Mat, Mat> >& MF_FDoG::getKerns() {
  return kerns;
}

inline Size MF_FDoG::getKernSize() {
  return kern_mf.size();
}
#endif /* End of MF_FDOG_H */
