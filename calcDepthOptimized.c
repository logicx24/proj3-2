// CS 61C Fall 2014 Project 3

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include "utils.h"
#include <stdio.h>
#include <string.h>

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"


#define ABS(x) (((x) < 0) ? (-(x)) : (x))
#define MAX(a, b) ( (a > b) ? a : b )
#define MIN(a, b) ( (a < b) ? a : b )

float displacementOptimized(int dx, int dy)
{

    return sqrt(dx * dx + dy * dy);
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, 
                        int featureWidth, int featureHeight, int maximumDisplacement) {

    float minimumSquaredDifference = -1;
    float minDisp = 0;
    int minimumDy;
    int minimumDx;
    float squaredDifference;
    float difference;

    memset(depth, 0, imageHeight*imageWidth*sizeof(float));
    #pragma omp parallel for
    for (int y = featureHeight; y < imageHeight-featureHeight; ++y) {
        for (int x = featureWidth; x < imageWidth-featureWidth; ++x)  {

            minDisp = 0;
            minimumSquaredDifference = -1;
            minimumDy = 0;
            minimumDx = 0;
            //#pragma omp parallel for
            for (int dx = MAX(-maximumDisplacement, featureWidth - x); dx <= MIN(maximumDisplacement, imageWidth - featureWidth - x - 1); ++dx) {
                for (int dy = MAX(-maximumDisplacement, featureHeight - y); dy <= MIN(maximumDisplacement, imageHeight - featureHeight - y - 1); ++dy) {

                    squaredDifference = 0;
                    __m128 store_vec = _mm_setzero_ps();
                    __m128 l_temp;
                    __m128 r_temp;
                    __m128 subbed;
                    __m128 mulled;
                    __m128 tail_vec = _mm_setzero_ps();
                    int boxX;
                    int topX = 0;
                    int bool1 = 1;

                    if ((2*featureWidth + 1) >= 8) {
                        //#pragma omp parallel for
                        for (boxX = -featureWidth; boxX <= featureWidth; boxX+=8) {
                            for (int boxY = -featureHeight; boxY <= featureHeight; ++boxY) {
                                
                                if (bool1 == 1 && boxX + 8 > featureWidth) {
                                    topX = boxX;
                                    bool1 = 0;
                                    //break;
                                }

                                l_temp = _mm_loadu_ps((left + ((y + boxY)*imageWidth + x + boxX)));
                                r_temp = _mm_loadu_ps((right + ((y + boxY + dy)*imageWidth + x + boxX + dx)));
                                subbed = _mm_sub_ps(l_temp, r_temp);
                                mulled = _mm_mul_ps(subbed, subbed);
                                store_vec = _mm_add_ps(store_vec, mulled);

                                l_temp = _mm_loadu_ps((left + ((y + boxY)*imageWidth + x + boxX + 4)));
                                r_temp = _mm_loadu_ps((right + ((y + boxY + dy)*imageWidth + x + boxX + 4 + dx)));
                                subbed = _mm_sub_ps(l_temp, r_temp);
                                mulled = _mm_mul_ps(subbed, subbed);
                                store_vec = _mm_add_ps(store_vec, mulled);

                            }
                        }
                        bool1 = 1
			            if (featureWidth - topX >= 4) {
		 	                for (boxX = topX; boxX <= featureWidth; boxX+=4) {
                                for (int boxY = -featureHeight; boxY <= featureHeight; ++boxY) {
                                
                                    if (bool1 == 1 && boxX + 4 > featureWidth) {
                                        bool1 = 0;
                                        topX = boxX;
                                        //break;
                                    }

                                    l_temp = _mm_loadu_ps((left + ((y + boxY)*imageWidth + x + boxX)));
                                    r_temp = _mm_loadu_ps((right + ((y + boxY + dy)*imageWidth + x + boxX + dx)));
                                    subbed = _mm_sub_ps(l_temp, r_temp);
                                    mulled = _mm_mul_ps(subbed, subbed);
                                    store_vec = _mm_add_ps(store_vec, mulled);

                                } 
                            }
			            }
                    }

      
                    else {
                        for (boxX = -featureWidth; boxX <= featureWidth; boxX+=4) {
                            for (int boxY = -featureHeight; boxY <= featureHeight; ++boxY) {
                                
                                if (bool1 && boxX + 4 > featureWidth) {
                                    bool1 == 0;
                                    topX = boxX;
                                    //break;
                                }

                                l_temp = _mm_loadu_ps((left + ((y + boxY)*imageWidth + x + boxX)));
                                r_temp = _mm_loadu_ps((right + ((y + boxY + dy)*imageWidth + x + boxX + dx)));
                                subbed = _mm_sub_ps(l_temp, r_temp);
                                mulled = _mm_mul_ps(subbed, subbed);
                                store_vec = _mm_add_ps(store_vec, mulled);

                            }
                        }
                    }
                    float r[4] = {0};

                    _mm_storeu_ps(r, store_vec);
                    squaredDifference += (r[0] + r[1] + r[2] + r[3]);
 
                    if (squaredDifference >= minimumSquaredDifference && minimumSquaredDifference != -1) 
                        continue;

                    if (featureWidth % 2 == 0) {
                        for (int boxY = -featureHeight; boxY <= featureHeight; ++boxY) {
                            difference = left[(y + boxY)*imageWidth + x + topX] - right[(y + boxY + dy)*imageWidth + x + topX + dx];
                            squaredDifference += difference * difference;
                        }
                    }
		   
                    else  {
                        for (int boxY = -featureHeight; boxY <= featureHeight; ++boxY) {
                            l_temp = _mm_loadu_ps((left + ((y + boxY)*imageWidth + x + topX)));
                            r_temp = _mm_loadu_ps((right + ((y + boxY + dy)*imageWidth + x + topX + dx)));
                            subbed = _mm_sub_ps(l_temp, r_temp);
                            mulled = _mm_mul_ps(subbed, subbed);
                            tail_vec = _mm_add_ps(tail_vec, mulled);
                        }

                        _mm_storeu_ps(r, tail_vec);
                        squaredDifference += (r[0] + r[1] + r[2]);
                    }




                    if ((minimumSquaredDifference == -1) || (minimumSquaredDifference > squaredDifference) || (((minimumSquaredDifference == squaredDifference) && (displacementOptimized(dx, dy) < minDisp)))) {

                        minimumSquaredDifference = squaredDifference;
                        minimumDx = dx;
                        minimumDy = dy;
                        minDisp = displacementOptimized(minimumDx, minimumDy);
                    }
                }
            }
            
           // depth[y*imageWidth + x] = minDisp; 
            depth[y*imageWidth+x] = (minimumSquaredDifference != -1 ? (maximumDisplacement == 0 ? 0 : minDisp) : 0);
        }
    }
}
