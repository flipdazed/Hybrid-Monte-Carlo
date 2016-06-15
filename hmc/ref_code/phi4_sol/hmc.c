#include "phi4.h"

/*
 *  File hmc.c
 *  Molecular dynamics algorithm for the phi4 theory
 *
 *  double hmc(act_params_t *apars, hmc_params_t *hpars)
 *       Hybrid Monte Carlo algorithm, starts from the global field phi
 *       The parameters of the action (kappa and lambda) are passed by
 *       the apars, the parameters of the algorithm by hpars, here
 *       we need the number of trajectories (ntraj), the trajectory length
 *       (tlength) and the number of steps per trajectory (nstep).
 *       calls update_md() ntraj times.
 *       Returns the acceptance rate.
 *
 *
 *  static double hamilton()
 *       computes the value of the HMC Hamiltonian H=mom^2/2+S(phi)
 *
 *  static void move_phi(double eps)
 *       one of the two elementary building blocks of the leapfrog
 *       phi <- phi + eps*mom
 *
 *  static void move_m(double eps)
 *       the other elementary building block of the leapfrog
 *       mom <- mom - eps*dS/dphi(phi)
 *
 *
 *  static void update_hmc(hmc_params_t* pars)
 *       The actual workhorse for the hmc routine.
 *       Does one trajectory: 
 *       momentum heatbath, leapfrog integration and 
 *       acceptance step
 *
 */



/* HMC momenta */
static double mom[V];
/* saved phi field for hmc */
static double phiold[V];
/* book keeping for acceptance rate */
static int accept, reject;
static double expdH;

static double kappa, lambda;

/**********************************************************************
 *     hamilton
 **********************************************************************/


static double hamilton()
{
    double act;
    int i;

    /* H=p^2/+S*/

    act=action();

    for (i=0;i<V;i++) act+=mom[i]*mom[i]/2;

    return act;
}


/********************************************************************** 
 *     move_phi()
 *     elementary leap frog step for the update of the phi field
 *     does phi <- phi+mom*eps
 **********************************************************************/
static void move_phi(double eps)
{
    int i;

    for (i=0;i<V;i++)
    {
	phi[i]+=mom[i]*eps;
    }

}

/********************************************************************** 
 *   move_m()
 *   elementary leap frog step for the update of the momenta
 *   does mom <- mom-eps*force
 **********************************************************************/

static void move_m( double eps)
{
    int i,mu;
    double phin, force;


    for (i=0;i<V;i++)
    {
	phin=0;
	for (mu=0;mu<2*D;mu++) phin+=phi[hop[i][mu]];

	force=2*kappa*phin-2*phi[i]-lambda*4*(phi[i]*phi[i]-1)*phi[i];
	mom[i]+=force*eps;
    }

}

/********************************************************************** 
 * update_hmc()
 * Do one trajectory with of the HMC
 **********************************************************************/

static void update_hmc(hmc_params_t* pars)
{
    int i, istep, nstep=pars->nstep;
    double eps=(pars->tlength)/(pars->nstep);
    double startH, endH, deltaH;
    double r;

    /*
     *  Action: Sum_x -2*kappa*sum_mu phi_x phi_{x+mu}+phi_x^2+lambda(phi_x^2-1)^2
     */

    /* refresh the momenta */
    gauss_rand(V,mom); 

    /* keep old phi field */
    for (i=0;i<V;i++) phiold[i]=phi[i];

    /* measure hamiltonian */
    startH=hamilton();

    /* do the trajectory */
    for (istep=0;istep<nstep;istep++)
    {
	move_phi(eps/2.);
	move_m  (eps);
	move_phi(eps/2.);
    }

    /* compute energy violation */
    endH=hamilton();
    deltaH=endH-startH;
    expdH+=exp(-deltaH);

    /* acceptance step */
    if (deltaH<0) {
	accept++;
    } else {
	ranlxd(&r,1);
	if (exp(-deltaH)>r) {
	    accept++;
	} else {
	    reject++;
	    for (i=0;i<V;i++) phi[i]=phiold[i];
	}
    }



}


/********************************************************************** 
 * hmc()
 * Does n trajectories of the HMC algorithm. Measurement after each
 * trajectory, the averaged measured values are printed out in fixed
 * intervals.
 **********************************************************************/
double hmc(act_params_t *apars, hmc_params_t *hpars)
{
    int isweep;

    lambda=apars->lambda;
    kappa =apars->kappa;

    accept=reject=0;

    expdH=0.0;
    for (isweep=0;isweep<hpars->ntraj;isweep++)
    {
	update_hmc(hpars);
	measure();
	if((isweep+1)%(hpars->naccu)==0) {
	    print_meas();
	    printf("EXPDH %e\n",expdH/hpars->naccu);
	    expdH=0.0;
	}
    }

    return ((double)accept)/((double)accept+reject);
}
