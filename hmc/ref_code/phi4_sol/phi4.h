#ifndef PHI4_H
#define PHI4_H
#include "stdio.h"
#include "stdlib.h"
#include "lattice.h"
#include "ranlxd.h"
#include "math.h"

void hopping(int h[V][2*D]);
void print_meas();
void measure();
void init_measure();
void gauss_rand(int n, double* rand);

/*data structure to store all the parameters of the algorithm*/
typedef struct {
               double tlength;  /*trajectory length*/
	       int nstep;       /*leapfrog steps per trajectory*/
               int ntherm ;     /*number of thermalization steps*/
	       int ntraj ;      /*number of trajectories after thermalization*/
	       int naccu;       /*binsize for printing out the measurements*/
              } hmc_params_t;

/*data structure to store all the parameters of the action*/
typedef struct {
              double kappa;
	      double lambda;
              } act_params_t;


double hmc(act_params_t *apara, hmc_params_t *hpara);
double action(void);
#endif
