/* 
 *    File measure.c
 *
 *    Routines for a blocking measurements of some simple observables.
 *    First init_measure is called to set counters and values to zero,
 *    then on call measure on a set of n configurations,
 *    print_meas() then prints the average values on the n configs
 *    and initializes the measurement process again.
 *     
 *    void init_measure()  
 *         Intialize the measurement process
 *
 *    void measure()
 *         Measures for the magnetization m: m^2/V and m^4/V as
 *         well as their correlators with the interaction term
 *         and this terms value
 *   
 *    void print_meas()
 *         Prints the average values of the above mentioned observables
 *         over the measurements since the last call to init_measure()
 *         and then calls this initialization function.
 *         Format:
 *         MEAS <No of measurements> <m/V> <m^2/V> <m^4/V> <W> <W m^2> <W m^4>
 */

#include "phi4.h"

static double M, m2, m4; /* m^2, m^4 */
static double W, Wm2, Wm4; /* W, W*m^2, W*m^4 */
static int nmeas;

void init_measure()
{
    M=m2=m4=0.;
    W=Wm2=Wm4=0.;
    nmeas=0;
}

void measure()
{
    int i, mu;
    double m, mm;
    double phin,w;


    m=0.0;
    w=0.0;
    for (i=0;i<V;i++){
	/*magnetization*/
	m+=phi[i];

	/*interaction term*/
	phin=0;
	for (mu=0;mu<D;mu++) phin+=phi[hop[i][mu]];

	w+=2.*phin*phi[i];

    }

    /* add to the averages */
    mm=m*m/V;
    /*powers of the magnetization*/
    M+=m/V;
    m2+=mm;
    m4+=mm*mm;

    /*interaction term and correlators*/
    W+=w;
    Wm2+=mm*w;
    Wm4+=mm*mm*w;

    nmeas++;
}

void print_meas()
{
    printf("MEAS %i %15.10e %15.10e %15.10e ",nmeas,M/nmeas,m2/nmeas,m4/nmeas);
    printf("%15.10e %15.10e %15.10e\n", W/nmeas,Wm2/nmeas,Wm4/nmeas);
    init_measure();
}

