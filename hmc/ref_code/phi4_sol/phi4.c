/* 
 *   File phi4.c
 *
 *   Contains the main program and a few other routines from which the
 *   code to simulate the phi**4 theory can be built. Routines for reading 
 *   in the main parameters of the action and the algorithm as  well as 
 *   the computation of the action are provided.
 *
 *   double action(void)
 *      This routine computes the action S[phi] for the global field phi in
 *      lattice.h and the parameters kappa and lambda from act_params.
 *      S = Sum_x [ -2*kappa*sum_mu phi_x phi_{x+mu}+phi_x^2+lambda(phi_x^2-1)^2 ]
 * 
 *
 *   static int get_val(FILE* fp, char *str, char* fmt,  void* val)
 *      Routine which reads one line from the input file.
 *      Format of the lines is <keyword> <value>.
 *      Checks if the keyword in string str matches,
 *      then gets the value according to the format in fmt
 *
 *     
 *   static int read_input(char *input)
 *      Parses the input file (format as specified in get_val)
 *      and prints the parameters onto the screen. Currently
 *      it reads the basic values for the action and also for the 
 *      future HMC and the seed of the random number generator.
 *
 *
 *    static double hamilton()
 *      Computes the HMC Hamiltonian p**2/2+S corresponding
 *      to the mom[] and phi[] fields
 */ 

#define CONTROL
#include "phi4.h"
#include "string.h"


/*  
 *  data structures to store all the parameters of the algorithm,
 *  and action defined in phi4.h
 *  seed for initialization of ranlux
 */

static hmc_params_t hmc_params;
static act_params_t act_params;
static int seed;


/**********************************************************************
 *    action
 **********************************************************************/

double action(void)
{
    int i,mu;
    double phin,S,phi2;
    double kappa =act_params.kappa;
    double lambda=act_params.lambda;

    S=0;

    /* loop over all sites */
    for (i=0;i<V;i++)
    {
	/*sum over neighbors in positive direction*/
	phin=0;
	for (mu=0;mu<D;mu++) phin+=phi[hop[i][mu]];

	phi2=phi[i]*phi[i];
	S+=-2*kappa*phin*phi[i]+phi2+lambda*(phi2-1.0)*(phi2-1.0);
    }

    return S;
}


/**********************************************************************
 *     get_val
 **********************************************************************/

static int get_val(FILE* fp, char *str, char* fmt,  void* val)
{
    char c[128];

    if(1!=fscanf(fp,"%s",c))
    {
	fprintf(stderr,"Error reading input file at %s\n",str);
	exit(1);
    }

    if(strcmp(str,c)!=0)
    {
	fprintf(stderr,"Error reading input file expected %s found %s\n",str,c);
	exit(1);
    }

    if(1!=fscanf(fp,fmt,val))
    {
	fprintf(stderr,"Error reading input file at %s\n",str);
	fprintf(stderr,"Cannot read value format %s\n",fmt);
	exit(1);
    }

    return 0;

}


/**********************************************************************
 *     read_input 
 **********************************************************************/


static int read_input(char *input)
{
    FILE* fp;

    fp=fopen(input,"r");
    if (fp==NULL) {
	fprintf(stderr, "Cannot open input file %s \n",input);
	exit(1);
    }

    get_val(fp, "kappa",       "%lf",&act_params.kappa  );
    get_val(fp, "lambda",      "%lf",&act_params.lambda );
    get_val(fp, "ntherm",      "%i" ,&hmc_params.ntherm );
    get_val(fp, "ntraj",       "%i" ,&hmc_params.ntraj  );
    get_val(fp, "traj_length", "%lf",&hmc_params.tlength);
    get_val(fp, "nstep",       "%i" ,&hmc_params.nstep  );
    get_val(fp, "seed",        "%i" ,&seed   );
    get_val(fp, "naccu",        "%i" ,&hmc_params.naccu   );


   
    printf("PARAMETERS\n");
    printf("L              %i\n", L);
    printf("DIM            %i\n", D);
    printf("kappa          %f\n", act_params.kappa);
    printf("lambda         %f\n", act_params.lambda);
    printf("ntherm         %i\n", hmc_params.ntherm);
    printf("ntraj          %i\n", hmc_params.ntraj);
    printf("traj_length    %f\n", hmc_params.tlength);
    printf("nstep          %i\n", hmc_params.nstep);
    printf("naccu          %i\n", hmc_params.naccu);
    printf("END PARAMETERS\n");

    return 0;
}
/**********************************************************************
 *     main
 **********************************************************************/

int main(int argc, char* argv[])
{
    double acc;

    if (argc != 2) {
	fprintf(stderr, "Number of arguments not correct\n");
	fprintf(stderr, "Usage: %s <infile> \n",argv[0]);
	exit(1);
    }

    /* get the parameters from the input file */
    read_input(argv[1]);

    /* initialize random number generator */
    rlxd_init(1,seed);

    /* initialize the nearest neighbor field */
    hopping(hop);

    /* initialize phi field */
    ranlxd(phi,V);

    /*do the updating*/

    acc=hmc(&act_params, &hmc_params);

    printf("ACCRATE %e\n",acc);


    return 0;
}

