/**
 *  @file MF-FDOG.h
 *  Purpose: Header file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#ifndef MF_FDOG_H
#define MF_FDOG_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <math.h> // log, sin, ...

class MF_FDoG {
  public:
    /** Constructor of MF_FDoG
     *  @param L Length of the neighborhood along the y-axis to smooth noise.
     *  @param s Variance of Gaussian kernels in banks.
     *  @param t Constant selecting whole Gaussian in kernels.
     */
     MF_FDog(int _L, int _s, int _t=3);
  
    /** Method applies kernel bank to input image and provides maximal response as result.
     *  @param src Reference to source image.
     *  @param dst Reference to output image.
     *  @param kerns Kernel bank to be applied.
     */
    void process(Mat &src, Mat &dst, vector<Mat> &kerns);
    
    /** Method calculates MF and FDoG kernel banks.
     */    
    void calcBanks();

  private:
    /** Method calculating kernel bank for MF or FDoG.
     *  @param kern Reference to kernel to be used.
     *  @param fdog If true, FDoG kernel bank is computed, otherwise MF kernel bank is computed.
     *  @return Bank of kernels.
     */    
    vector<Mat> calcBank(Mat &kern, bool fdog=false);
    
    /** Method calculating base kernel for MF or FDoG.
     *  @param fdog If true, kernel for first derivative of DoG. Otherwise, Gaussian matched filter is computed (default).
     *  @return Reference to computed kernel.
     */
    void calcKernel(Mat& kern, bool fdog=false);

    int L;    /**< Length of the neighborhood along the y-axis to smooth noise. */
    int s;  /**< Variance of the Gaussian. */
    int t;  /**< Constant, sigma scale to get to Gaussian width. */
    Mat kern_mf;  /**< Base kernel for Gaussian Matched Filter. */
    Mat kern_fdog; /**< Base kernel for FDoG Filter. */
    vector<Mat> kerns_mf; /**< Kernel bank for Gaussian Matched Filter. */
    int mf_size;
    vector<Mat> kerns_fdog; /**< Kernel bank for FDoG Filter. */
    int fdog_size;
}

#endif /* End of MF_FDOG_H */
