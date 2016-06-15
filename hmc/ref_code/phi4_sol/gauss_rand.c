#include <math.h>
#include "ranlxd.h"


/* 
 *    File gauss.c
 *
 *    void gauss_rand(int n, double* rand)
 *        generates n double precision gaussian random numbers
 *        Box-Muller procedure
 */
void gauss_rand(int n, double* rand)
{
   static double tpi;
   double tmp[2], r, phi;
   int i;
   static int called=0;

   if (!called) {
       tpi=8.*atan(1.);
       called=1;
   }

   i=0;
   while(i<n){

       /* two random numbers, flat distribution in [0,1) */
       ranlxd(tmp,2);

       /* compute polar coordinates: angle and radius */
       phi=tmp[0]*tpi;
       r  =sqrt(-2.*log(1.-tmp[1])); /* map second number [0,1) -> (0,1] */

       rand[i]=r*cos(phi); 
       i++;
       /*compute second only if requested */
       if (i<n) rand[i]=r*sin(phi); 
       i++;
   }
}

