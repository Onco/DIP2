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
    

  private:
    void calcKernels(bool fdog=false);
  
    int L;    /**< Length of the neighborhood along the y-axis to smooth noise. */
    int s;  /**< Variance of the Gaussian. */
    int t;  /**< Constant, sigma scale to get to Gaussian width. */
}

#endif /* End of MF_FDOG_H */
