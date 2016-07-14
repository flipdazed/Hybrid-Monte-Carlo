
/*******************************************************************************
 *
 * File analysis_smd2.c
 *
 * this file do the analysis of the data caming from smd2.c 
 *
 * to use the data file have to end with r#
 * 
 *the nubmer of replicas have to start from 1 otherwise there is segmentation faul
 * 
 *eg: test_r_1 test_r_2 test_r_1 test_r_3
 * 
 *
 * 
 *
 *******************************************************************************/

#define CONTROL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "lattice.h"
#include "utils.h"
#include "random.h"
#include "contract.h"
#include "towers.h"
#include "pseries.h"
#include "geometry.h"
#include "version.h"
#include "wflow.h"
#include "vfld.h"
#include "renorm.h"


static struct
{
   int nt,order;
   double *t;
} file_head;

static struct
{
   double ***EnK,***EnP,***EnQ;
   double ***G2z,***G2p,***G2q,***G4z;
} ddata;

static int nmeas,naccu,noms;
static int iroot,level,replicas,endian;
static char nbase[NAME_SIZE],**r;
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE],analysis_dir[NAME_SIZE];
static char log_file[NAME_SIZE],**dat_file[7];
static FILE *fin=NULL,*fdat=NULL;
static theory_t th;
long *size[7];
double **gammaFbb, *CbbF,*taubb_intF,*dtau,*dobs,*ddobs,*obs,*abb;
int *w;






static void alloc_ddata(void)
{
   int order;
   int i,it,nt;

   nt=file_head.nt;
   order=file_head.order;

   ddata.EnK=(double***) malloc((nt+1)*sizeof(double**));
   ddata.EnP=(double***) malloc((nt+1)*sizeof(double**));
   ddata.EnQ=(double***) malloc((nt+1)*sizeof(double**));
   ddata.G2z=(double***) malloc((nt+1)*sizeof(double**));
   ddata.G2p=(double***) malloc((nt+1)*sizeof(double**));
   ddata.G2q=(double***) malloc((nt+1)*sizeof(double**));
   ddata.G4z=(double***) malloc((nt+1)*sizeof(double**));

   for(it=0;it<=nt;it++)
   {
      ddata.EnK[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.EnP[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.EnQ[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.G2z[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.G2p[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.G2q[it]=(double**) malloc((order+1)*sizeof(double*));
      ddata.G4z[it]=(double**) malloc((order+1)*sizeof(double*));

      for(i=0;i<=order;i++)
      {
         ddata.EnK[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.EnP[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.EnQ[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.G2z[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.G2p[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.G2q[it][i]=(double*) calloc((order+1),sizeof(double));
         ddata.G4z[it][i]=(double*) calloc((order+1),sizeof(double));
      }
   }
}




static void read_ddata(double ***obs)
{
   int i,j,it;
   int iw,nt,order;
   double dstd[1];   

   iw=0;
   nt=file_head.nt;   
   order=file_head.order;

   for (it=0;it<=nt;it++)
   {
      for (i=0;i<=order;i++)
      {
         for (j=0;j<=order;j++)
         {
            dstd[0]=obs[it][i][j];
            dstd[0]/=(double)(naccu);
            obs[it][i][j]=0.0;

            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);

            iw+=fread(dstd,sizeof(double),1,fdat);
            obs[it][i][j]=dstd[0];
         }
      }
   }  
  
  error_root(iw!=((nt+1)*(order+1)*(order+1)),1,
             "write_data [smd2.c]","Incorrect write count");
}



static void read_theory_parms(void)
{
    int order;
   find_section(fin,"Theory");
   read_line(fin,"order","%d",&order);
   read_line(fin,"iroot","%d",&iroot);

   error(order<=0,1,"read_theory_parms [smd2.c]",
         "Parameter order must be positive");

   error((iroot!=0)&&(iroot!=1),1,"read_theory_parms [tm1.c]",
         "Parameter iroot must be 0 or 1");

   th=PHI4;
   file_head.order=order;
}


static void read_lat_parms(void)
{
   double kappa;

   find_section(fin,"Lattice parameters");
   read_line(fin,"z","%lf",&kappa);
   
   kappa=L*((double)L)/(kappa*kappa+2.*D*L*L);
   error(kappa>=(1.0/(2.0*(double)(D))),1,"read_lat_parms [smd2.c]",
         "For stability reasons kappa < 1/(2*D)");

   act_parms.kappa=kappa;
}


static void read_meas_parms(void)
{
   find_section(fin,"Measurements");
   read_line(fin,"nmeas","%d",&nmeas);
   read_line(fin,"naccu","%d",&naccu);
  
   error((nmeas<=0),1,"read_meas_parms [smd2.c]",
         "nmeas must be positive");

   error_root((naccu<1)||(nmeas<naccu)||((nmeas%naccu)!=0),1,
               "read_meas_parms [smd2.c]",
               "nmeas must be a multiple of naccu");
}


static void read_rng_parms(void)
{
   find_section(fin,"Random number generator");
   read_line(fin,"replicas","%d" ,&replicas );
   read_line(fin,"level","%d",&level);
}




static void read_wflow_parms(void)
{
   int nt,it,ie;
   double *t;
  
    
   find_section(fin,"Wilson flow");
   nt=count_tokens(fin,"c");

   t=malloc(2*nt*sizeof(*t));
   error(t==NULL,1,"read_wflow_parms [smd2.c]",
         "Unable to allocate times array");

   read_dprms(fin,"c",nt,t);
   
   for(it=0;it<nt;it++)
      t[it]=t[it]*t[it]*L*L/8.; 
   
   
   file_head.nt=nt;
   file_head.t=t;

   ie=0;

   for (it=0;it<nt;it++)
   {
      if (it==0)
         ie|=(t[it]<0.0);
      else
         ie|=(t[it]<t[it-1]);
   }

   error_root(ie!=0,1,"read_wflow_parms [smd2.c]",
              "Negative or not properly ordered flow times");
}


static void read_dirs(void)
{ 
   find_section(fin,"Run name");
   read_line(fin,"name","%s",nbase);
   
   
   find_section(fin,"Directories");
   read_line(fin,"log_dir","%s",log_dir);
   read_line(fin,"dat_dir","%s",dat_dir);
   sprintf(analysis_dir,"./analysis");
}


static void setup_files(void)
{	
  int i;
   FILE *ftest;

   check_dir(log_dir);
   error(name_size("%s/%s.smd2.log",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [smd2.c]","log_dir/<run_name> is too long");
   sprintf(log_file,"%s/%s.smd2.log",log_dir,nbase);
   ftest=fopen(log_file,"r");
  

for(i=1;i<=replicas;i++)
{  
  
  sprintf(dat_file[0][i],"%s/%s%s.smd2.EnK.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[1][i],"%s/%s%s.smd2.EnP.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[2][i],"%s/%s%s.smd2.EnQ.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[3][i],"%s/%s%s.smd2.G2z.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[4][i],"%s/%s%s.smd2.G2p.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[5][i],"%s/%s%s.smd2.G2q.dat",dat_dir,nbase,r[i]);
  sprintf(dat_file[6][i],"%s/%s%s.smd2.G4z.dat",dat_dir,nbase,r[i]);
  
   check_dir(dat_dir);
   error(name_size("%s/%s%s.smd2.xxx.dat",dat_dir,nbase,r[0])>=NAME_SIZE,1,
         "setup_files [smd2.c]","dat_dir/<run name> is too long");

   
   ftest=fopen(dat_file[0][i],"r");fclose(ftest);
   fdat=fopen(dat_file[0][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[0][i] = ftell(fdat);
   size[0][i]=(size[0][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[0][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   printf("numbers of nmeas/naccu=%lu\n",size[0][i]);
   fclose(fdat);printf("setup file %d\n",i);
   
   
    ftest=fopen(dat_file[1][i],"r");fclose(ftest);
  fdat=fopen(dat_file[1][i],"r");
  fseek(fdat, 0, SEEK_END);
  size[1][i] = ftell(fdat);
  size[1][i]=(size[1][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
  size[1][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
  fclose(fdat);

   
   ftest=fopen(dat_file[2][i],"r");fclose(ftest);
   fdat=fopen(dat_file[2][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[2][i] = ftell(fdat);
   size[2][i]=(size[2][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[2][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   fclose(fdat);
   
   ftest=fopen(dat_file[3][i],"r");fclose(ftest);
   fdat=fopen(dat_file[3][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[3][i] = ftell(fdat);
   size[3][i]=(size[3][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[3][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   fclose(fdat);
   
   ftest=fopen(dat_file[4][i],"r");fclose(ftest);
   fdat=fopen(dat_file[4][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[4][i] = ftell(fdat);
   size[4][i]=(size[4][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[4][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   fclose(fdat);
   
   ftest=fopen(dat_file[5][i],"r");fclose(ftest);
   fdat=fopen(dat_file[5][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[5][i] = ftell(fdat);
   size[5][i]=(size[5][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[5][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   fclose(fdat);
   
   ftest=fopen(dat_file[6][i],"r");fclose(ftest);
   fdat=fopen(dat_file[6][i],"r");
   fseek(fdat, 0, SEEK_END);
   size[6][i] = ftell(fdat);
   size[6][i]=(size[6][i]-2*sizeof(stdint_t)-file_head.nt*sizeof(double));
   size[6][i]/=sizeof(double)*(file_head.order+1)*(file_head.order+1)*(file_head.nt+1);
   fclose(fdat);
   
   error(size[0][i]!=size[1][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
   error(size[1][i]!=size[2][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
   error(size[2][i]!=size[3][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
   error(size[3][i]!=size[4][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
   error(size[4][i]!=size[5][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
   error(size[5][i]!=size[6][i],1,"setup_file", "dimension of the dat_file of the different observable  not the same"  );
}  
   ftest+=1;
      
}


static void read_infile(int argc,char *argv[])
{
   int ifile;
    int i,j;
   

   ifile=find_opt(argc,argv,"-i");
   error((ifile==0)||(ifile==(argc-1)),1,"read_infile [data_ispt3.c]",
         "Syntax: data_ispt3 -i <input file>");
   fin=freopen(argv[ifile+1],"r",stdin);
   error(fin==NULL,1,"read_infile [data_ispt3.c]",
         "Unable to open input file");

   read_theory_parms();
   read_lat_parms();
   read_meas_parms();
   read_rng_parms();
   if(noms==0) 
      read_wflow_parms();
   else
   {
      file_head.nt=0;
      file_head.t=NULL;
   }
   read_dirs();
     r=(char**) malloc(sizeof(char*)*(replicas+1));
   for(i=0;i<=replicas;i++)
   {
     r[i]=(char*) malloc(sizeof(NAME_SIZE)); /*printf("All work and no play makes Marco a dull boy\t");*/
     sprintf(r[i],"%d",i);
     printf("%s\n",r[i]);
   }
   
   for(i=0;i<7;i++)
   {
     dat_file[i]=(char**) malloc(sizeof(char*)*(replicas+1));
      for(j=0;j<=replicas;j++)
      dat_file[i][j]=(char*) malloc(NAME_SIZE);
   }

for(i=0;i<7;i++)   
  size[i]=(long*) malloc(sizeof(long)*(replicas+1));

   setup_files();

   fclose(fin);
}

 
 void print_info(void)
 { 
   int i,n;
     
   printf("\n");

   printf("Instantaneous Stochastic Perturbation Theory for phi^4 theory\n");
   printf("-----------------------------------------------\n\n");

   printf("Program version %s\n\n",ISPT_RELEASE);
   if (endian==LITTLE_ENDIAN)
      printf("The machine is little endian\n\n");
   else
      printf("The machine is big endian\n\n");
   printf("Theory:\n");
   printf("order          %i\n\n",  file_head.order);

   printf("Lattice parameters:\n");
   printf("DIM            %i\n",  D);
   printf("L              %lu\n", L);
   printf("V              %lu\n", V);
   printf("kappa          %.4e\n\n",act_parms.kappa);

   printf("Measurements:\n");
   printf("nmeas          %i\n",nmeas);
   printf("naccu          %i\n",naccu);
   printf("replicas           %i\n",replicas);
   printf("level          %i\n\n",level);

   printf("Solver:\n");
   if (ispt_parms.solver==CG)
   {
      printf("CG solver\n");
      printf("nmx         %i\n",ispt_parms.nmx);
      printf("res         %.4e\n\n",ispt_parms.res);
   }
   else if (ispt_parms.solver==FFT)
      printf("FFT solver\n\n");

   printf("Wilson flow:\n");
   printf("Flow times    ");

   for (i=0;i<file_head.nt;i++)
   {
            n=fdigits(file_head.t[i]);
            printf(" %.*f",IMAX(n,1),file_head.t[i]);
   }

  printf("\n\n");

  
}
void read_ddata_samples(int i,double ***obs)
{

   int nt,it,iw,ord;
   stdint_t istd[2];
   double *t,dstd[1];
   int  order;
   
   if(i==0)
   {
	order=file_head.order;
	
	if(endian==BIG_ENDIAN)
	    bswap_int(2,istd);

	iw=fread(istd,sizeof(stdint_t),2,fdat);
	nt=(int)istd[0];
	ord=(int)istd[1];
	error(nt!=file_head.nt,1,"read dat file", "file head nt(number of times) do not match infile");
	error(ord!=order,1,"read dat file", "file head order do not match infile");
	
	t=malloc(2*nt*sizeof(*t));
	
	for (it=0;it<nt;it++)
	{

	    if (endian==BIG_ENDIAN)
	      bswap_double(1,dstd);

	    iw+=fread(dstd,sizeof(double),1,fdat);
	    
	    t[it]=dstd[0];
	    error(t[it]!=file_head.t[it],1,"read dat file", "file head time %d do not match infile",it);
	}
	  
	  
	error_root(iw!=(2+nt),1,"write_file_head [smd2.c]",
		    "Incorrect write count");
   }
   
        read_ddata(obs);
       

}
 
static void load_dat(double ****G2z,double ****G2p, double ****G4z,double ****G2q,double ****EnK,double ****EnP,double ****EnQ)
{
  int k;
  int  order;
  int i,in,j,l;
  int i1=0;
 
  order=file_head.order;
  for(k=0;k<replicas;k++)
  {  
	printf("reading dat file %d\n",k+1);
   fdat=fopen(dat_file[0][k+1],"rb");
   error_root(fdat==NULL,1,"save_dat [smd2.c]",
              "Unable to open data file");
  

   for(i=0;i<size[0][k+1];i++)
   {    
     read_ddata_samples(i,ddata.EnK);   
      for (in=0;in<=file_head.nt;in++)
      {
	for (j=0;j<=order;j++)
	 {
	   for (l=0;l<=order;l++)
           EnK[i+i1][in][j][l]=ddata.EnK[in][j][l]; 
	  }
      }
   }
   fclose(fdat);
 
 
   fdat=fopen(dat_file[1][k+1],"rb");
   error_root(fdat==NULL,1,"save_dat [smd2.c]",
	      "Unable to open data file");
	      
  
   for(i=0;i<size[0][k+1];i++)
   {
    read_ddata_samples(i,ddata.EnP);  
      for (in=0;in<=file_head.nt;in++)
      {
	for (j=0;j<=order;j++)
	 {
	   for (l=0;l<=order;l++)
	     EnP[i+i1][in][j][l]=ddata.EnP[in][j][l];
	  }
      }
   }      
   fclose(fdat);

   fdat=fopen(dat_file[2][k+1],"rb");
   error_root(fdat==NULL,1,"save_dat [smd2.c]",
              "Unable to open data file");
   
  for(i=0;i<size[0][k+1];i++)
   {
    read_ddata_samples(i,ddata.EnQ);
      for (in=0;in<=file_head.nt;in++)
      {
	for (j=0;j<=order;j++)
	 {
	   for (l=0;l<=order;l++)
	     EnQ[i+i1][in][j][l]=ddata.EnQ[in][j][l];
	  }
      }
   }      
   fclose(fdat);
   
   
   fdat=fopen(dat_file[3][k+1],"rb");
   error_root(fdat==NULL,1,"save_dat [smd2.c]",
	      "Unable to open data file");
	      
	      for(i=0;i<size[0][k+1];i++)
	      {
		read_ddata_samples(i,ddata.G2z); 	      
		for (in=0;in<=file_head.nt;in++)
		{
		  for (j=0;j<=order;j++)
		  {
		    for (l=0;l<=order;l++)
		      G2z[i+i1][in][j][l]=ddata.G2z[in][j][l];
		  }
		}
	      }      
	      fclose(fdat);
	      
	      
	      
	      fdat=fopen(dat_file[4][k+1],"rb");
	      error_root(fdat==NULL,1,"save_dat [smd2.c]",
			 "Unable to open data file");
			 
			 for(i=0;i<size[0][k+1];i++)
			 {
			 read_ddata_samples(i,ddata.G2p); 	   
			   for (in=0;in<=file_head.nt;in++)
			   {
			     for (j=0;j<=order;j++)
			     {
			       for (l=0;l<=order;l++)
				 G2p[i+i1][in][j][l]=ddata.G2p[in][j][l];
			     }
			   }
			 }      
			 fclose(fdat);
			 
			 fdat=fopen(dat_file[5][k+1],"rb");
			 error_root(fdat==NULL,1,"save_dat [smd2.c]",
				    "Unable to open data file");
				    
				    for(i=0;i<size[0][k+1];i++)
				    {
				    read_ddata_samples(i,ddata.G2q); 	      
				      for (in=0;in<=file_head.nt;in++)
				      {
					for (j=0;j<=order;j++)
					{
					  for (l=0;l<=order;l++)
					    G2q[i+i1][in][j][l]=ddata.G2q[in][j][l];
					}
				      }
				    }      
				    fclose(fdat);
				    
				    fdat=fopen(dat_file[6][k+1],"rb");
				    error_root(fdat==NULL,1,"save_dat [smd2.c]",
					       "Unable to open data file");
					       
					       for(i=0;i<size[0][k+1];i++)
					       {
						read_ddata_samples(i,ddata.G4z); 	        
						 for (in=0;in<=file_head.nt;in++)
						 {
						   for (j=0;j<=order;j++)
						   {
						     for (l=0;l<=order;l++)
						       G4z[i+i1][in][j][l]=ddata.G4z[in][j][l];
						   }
						 }
					       }      
					       fclose(fdat);
					       
		       i1+=size[0][k+1];
   
  }
}
/***************************
 *
 * *************************/

double **alloca_dpseries(double order)
{
double **r;
int i;
r=(double**) malloc(sizeof(double*)*(order+1));
for(i=0;i<=order;i++)
	r[i]=(double*) calloc(order+1,sizeof(double));

return r;
}

void scale_dpseries(int order,int ordm,int ordg,double s, double **in)
{
int i,j;
double **tmp;
tmp=mult_mon_dpseries(order,ordm,ordg,s,in);
for(i=0;i<=order;i++)
  for(j=0;j<=order;j++)
       in[i][j]=tmp[i][j];
  free_dpseries(order,tmp);
}
void scale_pseries(int order,int ordg,double mon, double *in)
{
int i;
double *tmp;
tmp=mult_mon_pseries(order,ordg,mon,in); 

for(i=0;i<=order;i++)
      in[i]=tmp[i];

free(tmp);
}


double **Gamma(int t, int var, int order, int rep,int nconf, double *a, double *bba)
{
  double **r;
  int i0,i1,i2,i3,alpha,N;
  
  alpha=(order+1)*var;
  N=alpha*rep;
  r=(double**) malloc(sizeof(double*)*(alpha));
  for(i0=0;i0<alpha;i0++)
    r[i0]=(double*) calloc(alpha,sizeof(double));
for(i0=0;i0<alpha;i0++)
  for(i1=0;i1<alpha;i1++)
    for(i2=0;i2<rep;i2++)
      for(i3=0;i3<(nconf-t);i3++)
            r[i0][i1]+= (a[i0+i2*alpha+i3*N]-bba[i0])*(a[i1+i2*alpha+(i3+t)*N]-bba[i1]);
     if(t==0) printf("r[0][0]=%f\n",r[0][0]);  
          
    
          /*printf("a-bba=%f\n",a[0]-bba[0]);*/
for(i0=0;i0<alpha;i0++)
  for(i1=0;i1<alpha;i1++)
	    r[i0][i1]/=((double)(rep*nconf-rep*t));
     
return r;
}
double *function1(int var, int order,int flow ,double *ah)
{
double *r,p;
int i;
double ***tmp,**tmp2,*dm,**m4;
int j,j1;

p=2.*sin( 4.*atan(1.)/((double)L) );
p=p*p;
r=(double*) calloc(order+1,sizeof(double));
tmp=(double***) malloc(3*sizeof(double**));
for(i=0;i<3;i++)
{/* bba[i]=(double*) calloc(order+1,sizeof(double));*/
tmp[i]=alloca_dpseries(order);
}


for(j=0;j<3;j++)
for(j1=0;j1<=order;j1++)
for(i=0;i<=order;i++)
{ /*if(i!=1)  r[i]=ah[i*var]*ah[2]/ah[1];
    else     r[i]=ah[i*var];*/
        tmp[j][j1][i]=ah[j+j1*3+i*var];      /*  printf("aaaaaaaaaaaaaaaaaaaaaaa\n");*/

}
           
	   m4=invert_dpseries(order,tmp[1]);
	   tmp2=mult_dpseries(order,tmp[0],m4);free_dpseries(order,m4);
		    tmp2[0][0]-=1.;
	       m4=invert_dpseries(order,tmp2);free_dpseries(order,tmp2);
	       
	   scale_dpseries(order,0,0,p,m4);
	  
	   dm=massren(order,m4);
	 
	   


tmp2=mult_dpseries(order,tmp[0],m4); free_dpseries(order,m4);
m4=mult_dpseries(order,tmp2,tmp2);      free_dpseries(order,tmp2);
tmp2=invert_dpseries(order,m4);       free_dpseries(order,m4);
m4=mult_dpseries(order,tmp2,tmp[2]); free_dpseries(order,tmp2);
/*
sub_pseries(order,bba[0],bba[1],r);
tmp1=invert_pseries(order,r);		free(r);
r=mult_pseries(order,bba[1],tmp1);	free(tmp1);
scale_pseries(order,0,p,r);		
tmp1=mult_pseries(order,bba[0],r);	free(r);
r=invert_pseries(order,tmp1);		free(tmp1);
tmp1=pow_pseries(order,r,2); 		free(r);
	 
r=mult_pseries(order,bba[2],tmp1); 	free(tmp1);
*/

free(r);  r=reduce_dps2ps(order,dm,m4); free_dpseries(order,m4);

scale_pseries(order,0,1./((double)V),r);
if(flow!=0) scale_pseries(order,0,((double)(file_head.t[flow-1]*file_head.t[flow-1])),r);
free(dm);
return r;
}


double **barf( int var, int order, int rep,int nconf,int flow, double *bba,double **ga)
{
double **r,*tmp,*tmp1;
double *h,*ah;
int i,N,alpha;

N=rep*nconf;
alpha=(order+1)*var;
ah=(double*) malloc(alpha*sizeof(double));
h=(double*) malloc(alpha*sizeof(double));
r=(double**) malloc(alpha*sizeof(double*));
for(i=0;i<alpha;i++)
{
  r[i]=(double*) calloc(order+1,sizeof(double));
  h[i]=sqrt(   ga[i][i]/((double)N*4.)  );
  ah[i]=bba[i];
}

for(i=0;i<alpha;i++)
{
  ah[i]+=h[i];
  tmp1=function1(var,order,flow,ah);
  ah[i]-=h[i];ah[i]-=h[i]; 
  tmp=function1(var,order,flow,ah);
  sub_pseries(order,tmp1,tmp,r[i]);
  free(tmp);free(tmp1);ah[i]+=h[i];
  scale_pseries(order,0,1./(2.*h[i]),r[i]  );
}

free(h);free(ah);
return r;
}


void mean_value(int var, int order,int rep, int nconf,int flow,double *a)
{
    int i0,i1,i2,alpha,imax;
    double **ab,N;
    double *Fbb,*Fb,*tmp;
    
    alpha=(order+1)*var;
    N=nconf*rep;
    imax=alpha*rep;
    for(i0=0;i0<alpha;i0++)
        abb[i0]=0;
   
    ab=(double**) malloc(rep*sizeof(double*));
    for(i1=0;i1<rep;i1++)
        ab[i1]=(double*) calloc(alpha,sizeof(double));
    
    Fb=(double*) calloc(order+1,sizeof(double));
    
    for(i0=0;i0<alpha;i0++)
        for(i1=0;i1<rep;i1++)
            for(i2=0;i2<nconf;i2++)
	     ab[i1][i0]+=a[i0+i1*alpha+i2*imax];   
	
	printf("nconf=%d\t replicas=%d\t alpha=%d\n",nconf,rep,alpha);
	   
    for(i0=0;i0<alpha;i0++)
        for(i1=0;i1<rep;i1++)
            abb[i0]+=ab[i1][i0];
    
    for(i0=0;i0<alpha;i0++)
        for(i1=0;i1<rep;i1++)
            ab[i1][i0]/=(double)nconf; printf("ab[1][0]=%f\trep=%d\n",ab[1][0],rep);
    for(i0=0;i0<alpha;i0++)        
	abb[i0]/=(double) (((double) nconf)*rep); 
    
   
    Fbb=function1(var,order,flow,abb);    
    printf("Fbb=%.15e\t",Fbb[3]);
    if(rep==1)   for(i0=0;i0<=order;i0++) obs[i0]=Fbb[i0];
    else
    {    
        for(i1=0;i1<rep;i1++)
        {
 printf(" ok untill here  replice %d\n",i1);
            tmp=function1(var,order,flow,ab[i1]);
            scale_pseries(order,0,nconf,tmp);
            add_pseries(order,tmp,Fb,Fb);
        }
        scale_pseries(order,0,1./((double)N),Fb);
            printf("\nFb=%.15e\n",Fb[0]);
        for(i0=0;i0<=order;i0++)
            obs[i0]=(((double)rep)*Fbb[i0]-Fb[i0])/(((double)rep)-1.);
    }
        
    free_dpseries(rep-1,ab);
}

double *gammaf( int var, int order,double **ga,double **fa)
{
int i,j,k,alpha;
double *r;
r=(double*) calloc(order+1,sizeof(double));
alpha=var*(order+1);

for(i=0;i<=order;i++)
  for(j=0;j<alpha;j++)
    for(k=0;k<alpha;k++)
      r[i]+=fa[j][i]*fa[k][i]*ga[j][k];
   
return r;
}

void windowing(int var,int order, int rep, int nconf, int flow,double *a, double *bba)
{
    double **fbba,**tmp,*g,*tau,Caa=0;
    int count=0, i,j,N,alpha;
    double S=1.5;
    
    alpha=(order+1)*var;
    /*Caa=(double*) malloc((order+1)*sizeof(double));
    for(i=0;i<=order;i++)
        Caa=(double) calloc(alpha,sizeof(double));
      */      
    g=(double*) calloc(order+1,sizeof(double));
    tau=(double*) calloc(order+1,sizeof(double));
    
    N=rep*nconf;
      
     for(i=0;i<=order;i++)
    {
        CbbF[i]=0;
        w[i]=0;
    }
    
    tmp=Gamma(0,  var,  order,  rep, nconf, a,  bba);
    fbba=barf(  var,  order,  rep, nconf, flow, bba, tmp);
    gammaFbb[0]=gammaf(var,order,tmp,fbba);
    add_pseries(order,CbbF,gammaFbb[0],CbbF);
    Caa+=tmp[0][0];
    free_dpseries(alpha-1,tmp);
    
    for(i=1;i<nconf;i++)
    {   
      
        tmp=Gamma(i,  var,  order,  rep, nconf, a,  bba);
        gammaFbb[i]=gammaf(var,order,tmp,fbba);
         if(w[0]==0)  Caa+=2*tmp[0][0];
        free_dpseries(alpha-1,tmp);
        for(j=0;j<=order;j++)
        {   
            if(w[j]==0)
            {
                CbbF[j]+=2*gammaFbb[i][j];
                taubb_intF[j]=CbbF[j]/(2.*gammaFbb[0][j]);
                tau[j]=S/(  log( (2.*taubb_intF[j]+1)/(2.*taubb_intF[j]-1)  ));
                g[j]=exp(-((double)i)/tau[j])- (tau[j]/ (sqrt((double)(i*N) ))  );
                if(g[j]<0)
                {  count++;  w[j]=i; }
                
            }
            /*if(j==0) printf("gammaFbb[%d]=%0.10f\n",i,gammaFbb[i][0]);*/
             if(count==order+1) break;
        }
        free(gammaFbb[i]);
        if(count==order+1) break;
    }
    free_dpseries(alpha-1,fbba);
    free(g);free(tau);

        for(j=0;j<=order;j++)
        {   
            if(w[j]==0)
            {
		printf("Windowing condition failed up to W = %d",nconf-1);
                w[j]=nconf-1;
	    }
        }
    for(j=0;j<=order;j++)
    {   
        
        gammaFbb[0][j]+=CbbF[j]/N;
        CbbF[j]+=CbbF[j]*(2.*w[j]+1)/N;
        taubb_intF[j]=CbbF[j]/(2.*gammaFbb[0][j]);
    }
    free(gammaFbb[0]);
    printf("\n");
   
    
}

/*void deleting_bias(int var,int order, int rep, int nconf, double *a, double *bba,int w)
{
    
  tmp=Gamma(0,  var,  order,  rep, nconf, a,  bba);
    fbba=barf_opt(  var,  order,  rep, nconf,  bba, tmp);
    gammaFbb[0]=gammaf(var,order,tmp,fbba);   
}
*/

void return_answer( int var, int order ,int rep, int nconf)
{
    int i,N;
    
    N=rep*nconf;
    for(i=0;i<=order;i++)
    {
        dobs[i]=CbbF[i]/((double)N);
        dobs[i]=sqrt(dobs[i]);
       
        ddobs[i]=dobs[i]*sqrt((w[i]+0.5)/N);
        dtau[i]=sqrt( (w[i]+0.5-taubb_intF[i])/ ((double)N) )*2.* taubb_intF[i] ;
           
    }
    
}

void printing(int order,int flow)
{
    int i;
    if(flow==0)   printf("flow time=0.00000\n");
    else          printf("flow time=%f\n",file_head.t[flow-1]);
    
    for(i=0;i<=order;i++)
    printf("w[%d]=%d \t",i,w[i]);
    printf("\n");
    
    for(i=0;i<=order;i++)
    printf("obs[%d]=%.15e\t",i,obs[i]);
    printf("\n");
    
    for(i=0;i<=order;i++)
        printf("dobs[%d]=%.15e\t",i,dobs[i]);
    printf("\n");
    
    for(i=0;i<=order;i++)
        printf("ddobs[%d]=%.15e\t",i,ddobs[i]);
    printf("\n");
    
    for(i=0;i<=order;i++)
        printf("tau[%d]=%.15e\t",i,taubb_intF[i]);
    printf("\n");
    
    for(i=0;i<=order;i++)
        printf("dtau[%d]=%.15e\t",i,dtau[i]);
    printf("\n");
}


double *function(double **bba)
{
int order=file_head.order;
double *r,/* *tmp1,*/p;

r=(double*) calloc((order+1),sizeof(double));
p=4*sin( 4.*atan(1.)/((double)L) )*sin( 4.*atan(1.)/((double)L) );
p+=0;
    
/*	 sub_pseries(order,bba[0],bba[1],r);
	 tmp1=invert_pseries(order,r);		free(r);
  	 r=mult_pseries(order,bba[1],tmp1);	free(tmp1);
	 scale_pseries(order,0,p,r);		

	 tmp1=mult_pseries(order,bba[0],r);	free(r);
	 r=invert_pseries(order,tmp1);		free(tmp1);
	 tmp1=pow_pseries(order,r,2); 		free(r);
	 
	 r=mult_pseries(order,bba[2],tmp1); 	free(tmp1);
	 scale_pseries(order,0,1./((double)V),r);
*/
add_pseries(order,bba[0],r,r);
return r;
}




double *F(double **ba)
{
int i;
int order=file_head.order;
double *r,*tmp1,p;

r=(double*) calloc((order+1),sizeof(double));
p=4*sin(4.0*atan(1.0)/((double)L))*sin(4.0*atan(1.0)/((double)L));
    
   

   	 sub_pseries(order,ba[0],ba[1],r);
	 tmp1=invert_pseries(order,r);		free(r);
  	 r=mult_pseries(order,ba[1],tmp1);	free(tmp1);
	 for(i=0;i<=order;i++)
		r[i]*=p;	 
	  
	 
	 tmp1=mult_pseries(order,ba[0],r);	free(r);
	 tmp1=ba[0];
	 r=invert_pseries(order,tmp1);		/*free(tmp1);*/
	 tmp1=pow_pseries(order,r,2); 		free(r);
	
         r=mult_pseries(order,ba[2],tmp1); 	free(tmp1);
	  scale_pseries(order,0,16.*16.,r);
return r;
}
/**********************************************************************
 *     main
 **********************************************************************/


int main(int argc,char *argv[])
{
   int i,j,tot,l,i1;
   double ***s,***js2pt,***jsp2pt,***js4pt,***jEnK,***jEnQ,***jEnP;
   int flow;
   double ****G2z,****G2p,****G4z,****G2q,****EnK,****EnP,****EnQ;
   double p;
   double **dm,**tmp,**m4;
  int order;
  double **bG2Z,**bG2p,**bEnQ,****tmp1,***tmp2,**bbtemp,***btemp;
  FILE *aaa;
  
  double *tmp3,*tmp4;
  int alpha,N,var,j1,j2,j3,j4,therm,cut;
  /*r=(char*) malloc(sizeof(char)*(replicas+1));*/

  /* r[0]="0";
   r[1]="1";
   r[2]="2";
   r[3]="3";
   r[4]="4";
   r[5]="5";
   r[6]="6";
   r[7]="7";
   r[8]="8";
   r[9]="9";*/
  endian=endianness();
  read_infile(argc,argv);
/*  for(i=1;i<=replicas;i++)
    size[0][i]=1000;*/
  order=file_head.order;

  alloc_ddata();
  
   /* initialize random number generator */
  tot=0;
 /*  for(i=1;i<=replicas;i++)
     for(j=0;j<=6;j++)
       size[j][i]=40;*/
   for(i=1;i<=replicas;i++)
    tot+=size[0][i];
  
   var=(order+1)*3;
   alpha=var*(order+1);
   N=replicas*alpha;
   
  p=4*sin(4.0*atan(1.0)/((double)L))*sin(4.0*atan(1.0)/((double)L));

  flow=file_head.nt; 
  gammaFbb=(double**) malloc(sizeof(double*)*size[0][1]);
  CbbF=(double*) calloc(order+1,sizeof(double));
  taubb_intF=(double*) calloc(order+1,sizeof(double));
  dtau=(double*) calloc(order+1,sizeof(double));
  dobs=(double*) calloc(order+1,sizeof(double));
  ddobs=(double*) calloc(order+1,sizeof(double));
  obs=(double*) calloc(order+1,sizeof(double));
  w=(int*) calloc(order+1,sizeof(int));
  abb=(double*) calloc(alpha,sizeof(double));
  
  
  s     =(double***) malloc(sizeof(double**)*(tot));
  js2pt =(double***) malloc(sizeof(double**)*(tot));
  js4pt =(double***) malloc(sizeof(double**)*(tot));
  jsp2pt=(double***) malloc(sizeof(double**)*(tot));
  jEnK =(double***) malloc(sizeof(double**)*(tot));
  jEnP =(double***) malloc(sizeof(double**)*(tot));
  jEnQ=(double***) malloc(sizeof(double**)*(tot));
  
  G2z     =(double****) malloc(sizeof(double***)*(tot));
  G2p     =(double****) malloc(sizeof(double***)*(tot));
  G2q     =(double****) malloc(sizeof(double***)*(tot));
  G4z     =(double****) malloc(sizeof(double***)*(tot));
  EnK     =(double****) malloc(sizeof(double***)*(tot));
  EnP     =(double****) malloc(sizeof(double***)*(tot));
  EnQ     =(double****) malloc(sizeof(double***)*(tot));
  
  
  dm    =(double**) malloc(sizeof(double*)*tot);
  
  for(i=0;i<tot;i++)
  {
   js2pt[i] =(double**) malloc((order+1)*sizeof(double));
   js4pt[i] =(double**) malloc((order+1)*sizeof(double));
   jsp2pt[i]=(double**) malloc((order+1)*sizeof(double));
   
   jEnK[i] =(double**) malloc((order+1)*sizeof(double));
   jEnP[i] =(double**) malloc((order+1)*sizeof(double));
   jEnQ[i]=(double**) malloc((order+1)*sizeof(double));
  
   
   s[i]=(double**) malloc((order+1)*sizeof(double*));
   for(l=0;l<=order;l++)
   {
    s[i][l]     =(double*) calloc((order+1),sizeof(double));
    js2pt[i][l]     =(double*) calloc((order+1),sizeof(double));
    js4pt[i][l]     =(double*) calloc((order+1),sizeof(double));
    jsp2pt[i][l]     =(double*) calloc((order+1),sizeof(double));
    jEnK[i][l]     =(double*) calloc((order+1),sizeof(double));
    jEnP[i][l]     =(double*) calloc((order+1),sizeof(double));
    jEnQ[i][l]     =(double*) calloc((order+1),sizeof(double));
   }
   G2z[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   G2p[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   G2q[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   G4z[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   EnK[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   EnP[i]      =(double***) malloc(sizeof(double**)*(flow+1));
   EnQ[i]      =(double***) malloc(sizeof(double**)*(flow+1)); 
   for(j=0;j<=flow;j++)
    {
	G2z[i][j]=(double**) malloc((order+1)*sizeof(double*));
	G2p[i][j]=(double**) malloc((order+1)*sizeof(double*));
	G2q[i][j]=(double**) malloc((order+1)*sizeof(double*));
	G4z[i][j]=(double**) malloc((order+1)*sizeof(double*));
	EnK[i][j]=(double**) malloc((order+1)*sizeof(double*));
	EnP[i][j]=(double**) malloc((order+1)*sizeof(double*));
	EnQ[i][j]=(double**) malloc((order+1)*sizeof(double*));
	
	for(l=0;l<=order;l++)
	{
	G2z[i][j][l]=(double*) calloc((order+1),sizeof(double));
	G2p[i][j][l]=(double*) calloc((order+1),sizeof(double));
	G2q[i][j][l]=(double*) calloc((order+1),sizeof(double));
	G4z[i][j][l]=(double*) calloc((order+1),sizeof(double));
	EnK[i][j][l]=(double*) calloc((order+1),sizeof(double));
	EnP[i][j][l]=(double*) calloc((order+1),sizeof(double));
	EnQ[i][j][l]=(double*) calloc((order+1),sizeof(double));
	}
    }
  }
    load_dat(G2z,G2p,G4z,G2q,EnK,EnP,EnQ);
  /* flog=freopen("./analysis/test" ,"wb",stdout);
  error(flog==NULL,1,"print_info [smd2.c]","Unable to open analysis_log file");
*/  print_info();
   /*for(i=0;i<tot;i++)
   printf("%f \t %f \t %f\n",G2z[i][0][0],G2p[i][0][0],G4z[i][0][0]);*/

/****************************************************************************************/

for(i1=0;i1<=flow;i1++)
{

	bG2Z  =alloca_dpseries(order);
	bG2p  =alloca_dpseries(order);
	bEnQ  =alloca_dpseries(order);
	
	tmp1=(double****) malloc(sizeof(double***)*(3));
	tmp2=(double***) malloc(sizeof(double**)*3);
	bbtemp=(double**) malloc(sizeof(double*)*3);
	for(j=0;j<3;j++)
	{	
                tmp2[j]=(double**) malloc(sizeof(double*)*tot);
		tmp1[j]=(double***) malloc(sizeof(double**)*tot);
		for(i=0;i<tot;i++)
		{
			tmp1[j][i]=alloca_dpseries(order);	
		}
	}



	btemp=(double***) malloc(sizeof(double**)*(replicas+1));
	for(j=1;j<=replicas;j++)      
	    btemp[j]=(double**) malloc(3*sizeof(double*));
	
	l=0;
	for(j=1;j<=replicas;j++)
	{
		for(i=0;i<size[0][j];i++)
		{ 
	  		add_dpseries(order,G2z[i+l][0],tmp1[0][i+l],tmp1[0][i+l]);   
	  		add_dpseries(order,G2p[i+l][0],tmp1[1][i+l],tmp1[1][i+l]);   
	  		add_dpseries(order,EnQ[i+l][i1],tmp1[2][i+l],tmp1[2][i+l]);
	               
		        add_dpseries(order,G2z[i+l][0],js2pt[j],js2pt[j]);
	  		add_dpseries(order,G2p[i+l][0],jsp2pt[j],jsp2pt[j]);
	                add_dpseries(order,G4z[i+l][0],js4pt[j],js4pt[j]);
	 	        add_dpseries(order,EnK[i+l][i1],jEnK[j],jEnK[j]);
   	 	        add_dpseries(order,EnP[i+l][i1],jEnP[j],jEnP[j]);
	 	        add_dpseries(order,EnQ[i+l][i1],jEnQ[j],jEnQ[j]);
		}
	l+=size[0][j];
	}
 
	for(j=1;j<=replicas;j++)
	{
		add_dpseries(order,js2pt[j],bG2Z,bG2Z);
		add_dpseries(order,jsp2pt[j],bG2p,bG2p);
		add_dpseries(order,jEnQ[j],bEnQ,bEnQ);
	
		scale_dpseries(order,0,0,1./((double)size[0][j]),js2pt[j]);
		scale_dpseries(order,0,0,1./((double)size[0][j]),jsp2pt[j]);
		scale_dpseries(order,0,0,1./((double)size[0][j]),jEnQ[j]);
        

	}
		scale_dpseries(order,0,0,1./((double)tot),bG2Z);
		scale_dpseries(order,0,0,1./((double)tot),bG2p);
		scale_dpseries(order,0,0,1./((double)tot),bEnQ);
/*************************************************************************
 * reducing dps2ps
 * ********************************/
 	 for(i=1;i<=replicas;i++)
	 {
		sub_dpseries(order,js2pt[i],jsp2pt[i],s[i]);
		tmp=invert_dpseries(order,s[i]);
		m4=mult_dpseries(order,jsp2pt[i],tmp);free_dpseries(order,tmp);
		scale_dpseries(order,0,0,p,m4);
		
		dm[i]=massren(order,m4); 
		free_dpseries(order,m4);
		
		btemp[i][0]=reduce_dps2ps(order,dm[i],js2pt[i]);
		btemp[i][1]=reduce_dps2ps(order,dm[i],jsp2pt[i]);
		btemp[i][2]=reduce_dps2ps(order,dm[i],jEnQ[i]);
	        free(dm[i]);
         }   
        
        
         sub_dpseries(order,bG2Z,bG2p,s[0]);
         tmp=invert_dpseries(order,s[0]);
         m4=mult_dpseries(order,bG2p,tmp);free_dpseries(order,tmp); 
	 scale_dpseries(order,0,0,p,m4);
	 
         dm[0]=massren(order,m4); 
	 free_dpseries(order,m4);
	
         bbtemp[0]=reduce_dps2ps(order,dm[0],bG2Z);
	 bbtemp[1]=reduce_dps2ps(order,dm[0],bG2p);
	 bbtemp[2]=reduce_dps2ps(order,dm[0],bEnQ);
	 
	/* free_dpseries(order,bG2Z);*/free_dpseries(order,bG2p);free_dpseries(order,bEnQ);
	 
	free(dm[0]);
	 
	 for(i=0;i<tot;i++)
	 {
	   sub_dpseries(order,tmp1[0][i],tmp1[1][i],s[i]);
	   tmp=invert_dpseries(order,s[i]);
	   m4=mult_dpseries(order,tmp1[1][i],tmp);free_dpseries(order,tmp);
	   scale_dpseries(order,0,0,p,m4);
	   
	   dm[i]=massren(order,m4); 
	   
	   tmp2[0][i]=reduce_dps2ps(order,dm[i],tmp1[0][i]);
	   tmp2[1][i]=reduce_dps2ps(order,dm[i],tmp1[1][i]);
	   tmp2[2][i]=reduce_dps2ps(order,dm[i],tmp1[2][i]);
	   
           free_dpseries(order,m4);
	  /**/
	 }
	
	 
	 tmp3=(double*) calloc(N*size[0][1],sizeof(double));
	 tmp4=(double*) calloc(alpha,sizeof(double));
/*	 for(j1=0;j1<var;j1++)
	   for(j2=0;j2<=order;j2++)
	     for(j3=0;j3<replicas;j3++)
	       for(j4=0;j4<size[0][1];j4++)
		    tmp3[j1+j2*var+j3*alpha+j4*N]=tmp2[j1][j4+j3*size[0][1]][j2];
	  for(j1=0;j1<var;j1++)
	    for(j2=0;j2<=order;j2++)
		tmp4[j1+j2*var]=bbtemp[j1][j2];
  */         
therm=0;
cut=0;cut++;
/*size[0][1]-=cut;*/
             for(j1=0;j1<var;j1++)
	   for(j2=0;j2<=order;j2++)
	     for(j3=0;j3<replicas;j3++)
	       for(j4=therm;j4<size[0][1];j4++)
		    tmp3[j1+j2*var+j3*alpha+(j4-therm)*N]=tmp1[j1%3][(j4)+j3*(size[0][1])][(j1-(j1%3))/3][j2];
	  
	    /* for(j3=0;j3<replicas;j3++)
	       for(j4=0;j4<size[0][1];j4++)
	          { printf("%f\n",tmp1[2][j4+j3*size[0][1]][0][0]); printf("problem in replicas %d configuration %d\n",j3,j4);}
*/
                for(i=0;i<tot;i++)
                { free_dpseries(order,tmp1[0][i]);free_dpseries(order,tmp1[1][i]);free_dpseries(order,tmp1[2][i]);}
 /*  var=1;       alpha=var*(order+1);N=var*(order+1)*replicas;
	 for(j1=0;j1<var;j1++)
	   for(j2=0;j2<=order;j2++)
	     for(j3=0;j3<replicas;j3++)
	       for(j4=0;j4<size[0][1];j4++)
		    tmp3[j1+j2*var+j3*alpha+j4*N]=G2z[j4+j3*size[0][1]][0][0][j2];
            for(j2=0;j2<=order;j2++)
		tmp4[j2]=bbtemp[0][j2];
   */         
            
	/* for(i=0;i<tot;i++)  
	 printf("a[%d]=%f\t bba=%f \n",i,tmp2[0][i][0],bbtemp[0][0]);*/
/****************************************************************************************************
*Test
*************************/
mean_value(var,order,replicas,size[0][1]-therm,i1,tmp3);

windowing(var,order,replicas,size[0][1]-therm,i1,tmp3,abb);
return_answer( var,order, replicas,  size[0][1]-therm);
aaa=freopen("./analysis/ZEnQ_smd2_tot.dat","ab",stdout);
printf("%d\t",(int)L);
printing(order,i1);
fflush(aaa);
freopen("/dev/tty","w",stdout);
printf("flow =%d \t  obs=%.15e\n*************new flow time********************\n",i1,obs[0]);

if(i1>0)if( sqrt(file_head.t[i1-1]*8)/((double)L)<0.201)if( sqrt(file_head.t[i1-1]*8)/((double)L)>0.199)
{
  aaa=freopen("./analysis/ZEnQ_smd2_obs.dat","ab",stdout);
  printf("%d\t",(int)L);
  for(i=0;i<=order;i++)
  {
    printf("%0.16e   %0.16e   %0.16e \t   ",obs[i],dobs[i], fabs( ddobs[i]) /* ,fabs(ddobs[i]/obs[i])+(dobs[i]/obs[i])*(dobs[i]/obs[i]) */ );
  }
  printf("\n");
  
  fflush(aaa);
  
  aaa=freopen("./analysis/ZEnQ_smd2_tau.dat","ab",stdout);
  printf("%d\t",(int)L);
  for(i=0;i<=order;i++)
  {
    printf(" %0.15e \t   %0.15e \t  ",taubb_intF[i], dtau[i] );
  }
  printf("\n");
  
  fflush(aaa);freopen("/dev/tty","w",stdout);


}
/*for(i=0;i<size[0][1];i++)
printf("tmp=%e\n",tmp3[i*(order+1)]);
*/


/***********************************************************/



/*****************************************************************************************/

/*
w=windowing( tot,tmp2, bbtemp);
for(j=0;j<=order;j++)
  printf("w[%d]=%d\n",j,w[j]);
for(i=0;i<100;i++)
{
  w[0]=i;
bf=tau_intw(tot,w,tmp2, bbtemp);
printf("tau_int[%d]=%f\n",i,bf[0]);

}*/
/*
for(j=0;j<=order;j++)
  {  printf("tau_int[%d]=%f\n",j,bf[j]);  }
*/
free(tmp3);free(tmp4);

        
        for(j=0;j<tot;j++)
	{
	  for(i=0;i<=order;i++)
	  {
	    for(l=0;l<=order;l++)
	    {   
	      
	      js2pt[j][i][l]=0;
	      jsp2pt[j][i][l]=0;
	      js4pt[j][i][l]=0;
	      jEnK[j][i][l]=0;
	      jEnP[j][i][l]=0;
	      jEnQ[j][i][l]=0;
	    }
	  }
	}
	
	free(bbtemp[0]);free(bbtemp[1]);free(bbtemp[2]);
	for(i=0;i<3;i++)
	  for(j=0;j<tot;j++)
	  {	free(tmp2[i][j]); 
		if(j>=1)  if(j<replicas)  free(btemp[j][i]); 
		if(i==0)   {free(dm[j]);   }
	  }
	
        
} 
  
 
  
  
   
   return EXIT_SUCCESS;
}
