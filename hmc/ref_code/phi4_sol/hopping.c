/*********************************************************
 * 
 *  File hopping.c
 *
 *  Initialization of the hopping field for a D dimensional
 *  lattice of size V=L**4
 *  The index of a point with coordinates (n_0,n_1,..,n_{d-1})
 *  is i=sum_k n_k L**k
 *  The index of its neighbor in positive direction nu 
 *  is hop[i][mu]
 *  In negative direction it is hop[i][D+mu]
 *
 ********************************************************/

#include "phi4.h"

void hopping(int hop[V][2*D] ){
    int x, y, Lk;
    int xk, k, dxk ;


    /* go through all the points*/
    for (x=0; x < V ; x++){
	Lk = V;
	y  = x;

	/* go through the components k*/
	for (k=D-1; k >= 0; k--){

	    Lk/=L;                        /* pow(L,k)      */
	    xk =y/Lk;                     /* kth component */
	    y  =y-xk*Lk;                  /* y<-y%Lk       */

	    /* forward */
	    if (xk<L-1) dxk = Lk;
	    else        dxk = Lk*(1-L);
	    hop[x][k] = x + dxk;

	    /* backward */
	    if (xk>0)   dxk = -Lk;
	    else        dxk = Lk*(L-1);
	    hop[x][k+D] = x + dxk;

	}
    }
} /* hopping */

