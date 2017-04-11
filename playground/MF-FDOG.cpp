/**
 *  @file MF-FDOG.cpp
 *  Purpose: Source file of the Gaussian Matched Filter implementation and its first derivative.
 *
 *  @author Ondrej Vavro
 *  @version 0.1
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <math.h> // log, sin, ...
#include "MF-FDOG.h"


/** Method calculating kernels for MF and FDoG.
 * @param argc Number of input arguments.
 * @param argv Pointer to the arguments.
 * @return Returns 0 on success, error code number otherwise.
 */
void calcKernels(bool fdog) {
	// dim_y = int(L)
    // dim_x = 2 * int(t * s)
    Mat arr = Mat::zeros(L, 2*t*s, )
    
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)

    # an un-natural way to set elements of the array
    # to their x coordinate. 
    # x's are actually columns, so the first dimension of the iterator is used
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    @vectorize(['float32(float32)'], target='cpu')
    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    @vectorize(['float32(float32)'], target='cpu')
    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
       kernel = k_fun_derivative(arr)

    # return the "convolution" kernel for filter2D
    return cv2.flip(kernel, -1) 
}
