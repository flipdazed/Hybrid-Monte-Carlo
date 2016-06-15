/* This program simulates SU(GROUP) lattice gauge fields with the
   simple Wilson action.

         Michael Creutz, mike@latticeguy.net

    The lattice dimensions are in "shape"
    Boundary conditions are periodic
    HITS = metropolis hits per link 
    monte() updates lattice with conventinal metropolis algorithm
    overrelax() does an overrelaxation algorithm
    loop(matrix * u, int x, int y) measures Wilson loops
 */

/* size and other parameters */
double beta=2.3; /* can also be set by first arguement to the program */
#define GROUP 2
#define DIM 4
#define SIZE 8
int shape[DIM]={SIZE,SIZE,SIZE,SIZE}; /* other shapes are allowed but 
 each dimension must be even for checkerboarding */
#define HITS 10

/* ------------------------------------ */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* class for handling group by group matrices */
class matrix {
public:
  double real[GROUP][GROUP];
  double imag[GROUP][GROUP];
  matrix& project(); /* projects onto the gauge group */
  void determinant(double& rez, double& imz);
  void printmatrix();
  matrix& operator= (double x);
  matrix& operator*= (double x);
};

matrix operator* (matrix&, matrix&);
matrix operator+ (matrix&, matrix&);
matrix operator- (matrix&, matrix&);
matrix conjugate(matrix);

/* global variables and temporaries */
int shift[DIM]; // shift to next site in given direction; initialize later
int nsites,nlinks,nplaquettes,vectorlength;
matrix *ulinks; /* for the main lattice */
matrix *table1,*table2; /* for the updating tables */
matrix *mtemp[5];
double *sold,*snew,smax=0.,pi;
int *accepted,*myindex,*myindex2, *parity, seed;
#define forvector for(iv=0;iv<vectorlength;iv++)
#define formatrix for(i=0;i<GROUP;i++)for(j=0;j<GROUP;j++)

/* prototype things */
void cleanup(const char *message);
/* functions on entire lattice */
void init();
double monte(matrix *u);
double overrelax(matrix *u);
void renorm(matrix *u);
void randomgauge(matrix *u);
/* functions on vectorlength matrices */
void vgroup(matrix *g);  // projects onto the group
void thirdrow(matrix& g); // for SU(3) only
void ranmat(matrix *g); // random matrices with a weight towards I
void vcopy(matrix *g1, matrix *g2);
void vprod(matrix *g1,matrix *g2,matrix *g3);
void vsum(matrix *g1,matrix *g2,matrix *g3);
void vtrace(matrix *g,double *s);
void vtprod(matrix *g1,matrix *g2,double *s);
void getlinks    (matrix *g ,matrix *lattice,int site,int link);
void getconjugate(matrix *g ,matrix *lattice,int site,int link);
void savelinks   (matrix *g ,matrix *lattice,int site,int link);
void staple      (matrix *g ,matrix *lattice,int site,int link);
double metro(matrix *g,matrix *trial,double bias);
void maketable();
void vtable(); // update matrix table
/* for measuring Wilson loops */
double loop(matrix * u, int x, int y);
/* utilities for periodic boundaries */
void makeindex(int n,int *ind);
int siteindex(int *x); /* gives a unique index to site located at x */
void split(int *x, int s); /* splits a site index into coordinates */
int vshift(int n, int *x); /* shifts site n by vector x */
int ishift( int n, int dir, int dist); /* site shifted dist in dir */

int main(int argc, char **argv) {
  int i,iter,count;
  double mytime;
  // let beta be changed if an argument present
  if (argc>1)
    beta=strtod(argv[1],NULL);
  init();
  printf("lattice size %d",shape[0]);
  for (i=1;i<DIM;i++)
    printf(" by %d",shape[i]);
  printf("\n vectorlength = %d\n",vectorlength);
  printf("group=SU(%d)   beta = %6.4f\n",GROUP,beta);
  printf("-----------------\n");

  /* experiment: standard Monte Carlo updating */
  printf("test monte\n");
  for (iter=0;iter<5;iter++) {
    mytime=clock();
    count=0;
    for (i=0;i<5;i++) {
      monte(ulinks);
      count++; 
    }
    renorm(ulinks);
    mytime=clock()-mytime;
    mytime=(1000000./(1.*count*nlinks*CLOCKS_PER_SEC))*mytime;
    printf("running at %g microseconds per link\n",mytime);
    loop(ulinks,2,2);
  } 

  printf("test overrelax\n");
  for (iter=0;iter<5;iter++) {
    mytime=clock();
    count=0;
    for (i=0;i<5;i++) {
      overrelax(ulinks);
      count++;
    } 
    renorm(ulinks);
    mytime=clock()-mytime;
    mytime=(1000000./(1.*count*nlinks*CLOCKS_PER_SEC))*mytime;
    printf("running at %g microseconds per link\n",mytime);    
  }
  cleanup("all done");
  return 0;
}

void init() {
  int i,iv,x[DIM];
  pi=4*atan(1.);
  srand48(1234);
  /* allocate space, set lattice to identity, initialize random table */
  nsites=1;
  for(i=0;i<DIM;i++){
    nsites*=shape[i];
    if (1&shape[i]) cleanup("bad dimensions");
  }
  nlinks=DIM*nsites;
  nplaquettes=DIM*(DIM-1)*nsites/2;
  vectorlength=nsites/2;
  /* allocate arrays */
  ulinks=new matrix[nlinks];
  parity=new int[nsites];
  table1=new matrix[vectorlength];
  table2=new matrix[vectorlength];
  for (i=0;i<5;i++)
    mtemp[i]=new matrix[vectorlength];
  sold=new double[vectorlength];
  snew=new double[vectorlength];
  accepted=new int[vectorlength];
  myindex=new int[vectorlength];
  myindex2=new int[vectorlength];

  /* initialize shift array for locating links */
  shift[0]=1;
  for (i=1;i<DIM;i++)
    shift[i]=shift[i-1]*shape[i-1];
  /* set starting links to identity matrix */
  for (iv=0;iv<nlinks;iv++)
    ulinks[iv]=1.0;
  /* set parity matrix for sites */
  for (iv=0;iv<nsites;iv++){
    split(x,iv);
    parity[iv]=0;
    for (i=0;i<DIM;i++)
      parity[iv] ^= x[i];
    parity[iv] &= 1;
  }
  maketable();
  printf("initialization done\n");
  return;
}

void maketable(){
  /* generate tables of vectorlength random matrices */
  int i,j,iv;
  matrix temporary1,temporary2;
  forvector{
    /* bias towards the identity */
    temporary1=beta/GROUP;
    temporary2=beta/GROUP;
    formatrix {
      temporary1.real[i][j]+=drand48()-0.5;
      temporary1.imag[i][j]+=drand48()-0.5;
      temporary2.real[i][j]+=drand48()-0.5;
      temporary2.imag[i][j]+=drand48()-0.5;
    } 
    table1[iv]=temporary1;
    table2[iv]=temporary2;
  }
  /* make into group elements */
  vgroup(table1);
  vgroup(table2);
  /* update table a few times */
  for (i=0;i<50;i++)
    vtable();
  return;
}

void vgroup(matrix *g)
/* subroutine to make group elements out of vectorlength matrices g */
{int iv;
  forvector
    g[iv].project();
  return;
}

double monte(matrix *lattice) {
  /* the basic metropolis update */
  double stot,acc,eds;
  int iv,iacc,color,link,hit;
  vtable(); /* update table */
  stot=eds=0.0;
  iacc=0;
  /* loop over checkerboard colors */
  for (color=0;color<2;color++) {
    /* loop over link dirs */
    for (link=0;link<DIM;link++) {
      staple(mtemp[4],lattice,color,link); /* get neighborhood */
      /* get old link and calculate action */
      getlinks(mtemp[0],lattice,color,link);
      vtprod(mtemp[0],mtemp[4],sold);
      /* loop over hits */
      for (hit=0;hit<HITS;hit++) {
	/* get random matrices */
	ranmat(mtemp[1]);
	/* find trial element and new action */
	vprod(mtemp[0],mtemp[1],mtemp[2]);
	vtprod(mtemp[2],mtemp[4],snew);
	eds+=metro(mtemp[0],mtemp[2],beta/(1.*GROUP)); /* metropolis step */
	forvector {
	  iacc=iacc+accepted[iv];
	  stot=stot+sold[iv];
	}  
      }
      savelinks(mtemp[0],lattice,color,link); /* save new links */
    }
  }
  stot=stot/(2.0*(DIM-1)*nlinks*GROUP*HITS);
  acc=iacc/(1.*nlinks*HITS);
  eds=eds/(2.*DIM*HITS);
  /* eds should fluctuate about unity when in equilibrium */
  printf("stot=%f, acc=%f, eds=%f\n",stot,acc,eds);
  return stot;
}

double overrelax(matrix *lattice) {
  double stot,acc,eds;
  int iv,iacc,color,link;
  if (GROUP>3) cleanup("overrelax needs GROUP<=3 or more temporaries");
  stot=eds=0.0;
  iacc=0;
  /* loop over colors */
  for (color=0;color<2;color++) 
    /* loop over link dirs */
    for (link=0;link<DIM;link++) {
      staple(mtemp[4],lattice,color,link); /* get neighborhood */
      /* get old link */
      getlinks(mtemp[0],lattice,color,link);
      /* find trial element and new action */
      vcopy(mtemp[4],mtemp[1]);
      vgroup(mtemp[1]);  
      vprod(mtemp[0],mtemp[1],mtemp[2]);
      vprod(mtemp[1],mtemp[2],mtemp[3]);
      forvector
	mtemp[2][iv]=conjugate(mtemp[3][iv]);
      vtprod(mtemp[0],mtemp[4],sold);
      vtprod(mtemp[2],mtemp[4],snew);
      eds+=metro(mtemp[0],mtemp[2],beta/(1.*GROUP)); /* metropolis step */
      forvector {
	iacc=iacc+accepted[iv];
	stot=stot+sold[iv];
      }  
      savelinks(mtemp[0],lattice,color,link); /* save new links */
    }
  stot=stot/(2.0*(DIM-1)*nlinks*GROUP);
  acc=iacc/(1.*nlinks);
  eds=eds/(2.*DIM);
  printf("stot=%f, acc=%f, eds=%f\n",stot,acc,eds);
  return stot;
}

void makeindex(int n,int *ind){
  /* generates a set of site labels starting at n for gathering links */
  /* loop over even parity sites and gather with shift n from them */
  int x[DIM],iv,site;
  split(x,n);
  site=0;
  forvector{
    while (parity[site]) site++;
    ind[iv]=vshift(site,x);
    site++;
  }
  return;
}

void getlinks(matrix *g,matrix *lattice,int site,int link){
  /* gather same color links into vector g starting at site */
  int iv,shift;
  makeindex(site,myindex);
  shift=nsites*link;
  forvector
    g[iv]=lattice[myindex[iv]+shift];
  return;
}

void getconjugate(matrix *g,matrix *lattice,int site,int link){
  /* gather conjugate links into vector g */
  int iv,shift;
  makeindex(site,myindex);
  shift=nsites*link;
  forvector
    g[iv]=conjugate(lattice[myindex[iv]+shift]);
  return;
}

void savelinks(matrix *g,matrix *lattice,int site,int link){
  /* scatter alternate links from vector g */
  int iv,shift;
  makeindex(site,myindex);
  shift=nsites*link;
  forvector
    lattice[myindex[iv]+shift]=g[iv];
  return;
}

void vprod(matrix *g1,matrix *g2,matrix *g3) {
  /* set g3 to the matrix product of g1 and g2, vectorlength times */
  int iv;
  forvector
    g3[iv]=g1[iv]*g2[iv];
  return;
}

void vsum(matrix *g1,matrix *g2,matrix *g3) {
  /* matrix sum of g1 and g2 to g3, vectorlength times */
  int i,j,iv;
  /*   slightly faster writing this out over:
       forvector
       g3[iv]=g1[iv]+g2[iv];
  */  
  formatrix 
    forvector {
    g3[iv].real[i][j]=g1[iv].real[i][j]+g2[iv].real[i][j];
    g3[iv].imag[i][j]=g1[iv].imag[i][j]+g2[iv].imag[i][j];
  }
  return;
}

void vcopy(matrix *g1, matrix *g2) {
  /* matrix copy of g1 to g2, vectorlength times */
  /*
  int iv;
  forvector {
    g2[iv]=g1[iv];
  }
  */
  memcpy((void *) g2, (void *) g1, vectorlength*sizeof(matrix));
  return;
}

void vtprod(matrix *g1,matrix *g2,double *s) {
  /* real trace of product g1 and g2 to s, vectorlength times */
  int i,j,iv;
  forvector s[iv]=0.0;
  formatrix 
    forvector  
    s[iv]+=g1[iv].real[i][j]*g2[iv].real[j][i]
    -g1[iv].imag[i][j]*g2[iv].imag[j][i];   
  return;
}

void vtrace(matrix *g,double *s) {
  /* real trace of g to s, vectorlength times */
  int i,iv;
  forvector s[iv]=g[iv].real[0][0];
  for (i=1;i<GROUP;i++) 
    forvector s[iv]+=g[iv].real[i][i];
  return;
}

void renorm(matrix *l) { 
  /* project whole lattice into group; call from time to time 
     to keep down drift from floating point errors */
  int iv,octant,link;
  /* loop over lattice octants */
  for (octant=0;octant<2*DIM;octant++) {
    link=octant*vectorlength;
    forvector
      l[link+iv].project();
  }
  return;
}

void vtable() /* update matrix table */
{/* shuffle table 1  into a */
  /* the random inversion from ranmat is important! */
  ranmat(mtemp[0]);
  /* multiply table 2 by a into table 1 for trial change */
  vprod(table2,mtemp[0],table1);
  /* metropolis select new table 2 */
  vtrace(table2,sold);
  vtrace(table1,snew);
  metro(table2,table1,6*beta/GROUP);  
  /* switch table 1 and 2 */
  vcopy(table2,table1);
  vcopy(mtemp[0],table2);
  vgroup(table1);
  return;
}
 
void staple(matrix *st, matrix *lat,int site,int link) {
  /* This subroutine calculates a vector of matrices interacting with
     with links using Wilson action.  The lattice is in lat and the
     result is placed in st.  The first three matrix vectors mtemp[0],
     mtemp[1], and mtemp[2], are used; so st should not be there and these
     shouldn't be used until after staple is done.  
     links and sites labeled as

     2--link2--x
     link3     link1
     0--link --1
     link6     link4
     5--link5--4
  */
  int iv,link1,site1,site2,site4,site5;
  forvector 
    st[iv]=0.0;
  site1=ishift(site,link,1);
  /* loop over planes */
  for (link1=0;link1<DIM;link1++)
    if (link1!=link) {
      site2=ishift(site ,link1, 1);
      site4=ishift(site1,link1,-1);
      site5=ishift(site ,link1,-1);
      /* top of staple */
      getlinks(mtemp[0],lat,site1,link1);
      getconjugate(mtemp[1],lat,site2,link);
      vprod(mtemp[0],mtemp[1],mtemp[2]);
      getconjugate(mtemp[0],lat,site,link1);
      vprod(mtemp[2],mtemp[0],mtemp[1]);
      vsum(st,mtemp[1],st);
      /* bottom of staple */
      getconjugate(mtemp[0],lat,site4,link1);
      getconjugate(mtemp[1],lat,site5,link );
      vprod(mtemp[0],mtemp[1],mtemp[2]);
      getlinks(mtemp[0],lat,site5,link1);
      vprod(mtemp[2],mtemp[0],mtemp[1]);
      vsum(st,mtemp[1],st);
    }
  return;
}
 
double metro(matrix *old,matrix *trial,double bias) {
  /* accept new for old using metropolis algorithm
     return average exponential of action change, this should fluctuate
     about unity when in equilibrium
     bias multiplies actions in exponential (i.e. beta/GROUP)  
     accepted changes returned in accepted
     actions passed in global variables sold and snew */
  int iv;
  double expdeltas=0.0,temp;
  forvector {
    temp=exp(bias*(snew[iv]-sold[iv]));
    expdeltas=expdeltas+temp;
    accepted[iv]=(drand48()<temp);
  }
  /* Accept changes */
  forvector
    if (accepted[iv]) {
      sold[iv]=snew[iv];
      old[iv]=trial[iv];
    }
  return expdeltas/vectorlength;
}   
 
void ranmat(matrix *g) {
  /* randomly shift table1, randomly invert, and put in g */
  int iv,index;
  index=(int) (vectorlength*drand48());
  forvector {
    if (index>=vectorlength) index-=vectorlength;
    if (drand48()<0.5)
      g[iv]=table1[index];
    else
      g[iv]=conjugate(table1[index]);
    index++;
  }
  return;
}
 
void cleanup(const char *message) {
  int i;
  printf("%s\n",message);
  delete[] ulinks;
  for (i=0;i<5;i++)
    delete[] mtemp[i];
  delete[] parity;
  delete[] table1;
  delete[] table2;
  delete[] sold;
  delete[] snew;
  delete[] accepted;
  delete[] myindex;
  delete[] myindex2;
  exit(0);
}

void thirdrow(matrix& g) {
  /* for su(3) construct third row from first two */
  int i,j,k;
  for (i=0;i<3;i++) {
    j=(i+1)%3;       
    k=(i+2)%3;
    g.real[2][i]= g.real[0][j]*g.real[1][k]
      -g.imag[0][j]*g.imag[1][k]
      -g.real[1][j]*g.real[0][k]
      +g.imag[1][j]*g.imag[0][k];
    g.imag[2][i]=-g.real[0][j]*g.imag[1][k]
      -g.imag[0][j]*g.real[1][k]
      +g.real[1][j]*g.imag[0][k]
      +g.imag[1][j]*g.real[0][k];
  }
  return; 
}

void randomgauge(matrix * u) {
  /* make a semi-random gauge transformation on lattice u */
  /* the gauge transform is weighted toward the identity as in the table */
  int iv,color,link;
  vtable(); /* update table */
  /* loop over checkerboard colors */
  for (color=0;color<2;color++) {
    /* get random matrices for gauge transforming*/
    ranmat(mtemp[1]);
    for (link=0;link<DIM;link++) {
      getlinks(mtemp[0],u,color,link);
      vprod(mtemp[1],mtemp[0],mtemp[2]);
      savelinks(mtemp[2],u,color,link);
    }
    forvector
      mtemp[1][iv]=conjugate(mtemp[1][iv]);
    for (link=0;link<DIM;link++) {
      /* get negative side links */
      getlinks(mtemp[0],u,ishift(color,link,-1),link);
      vprod(mtemp[0],mtemp[1],mtemp[2]);
      savelinks(mtemp[2],u,ishift(color,link,-1),link);
    }
  }
  return;
}

double loop(matrix * u, int x, int y){
  /* calculate rectangular wilson loops */
  int i,color,link1,link2,iv,corner,count=0;
  double result=0.;
  for(color=0;color<2;color++)
    for (link1=0;link1<DIM;link1++)
      for (link2=(x==y)*(link1+1);link2<DIM;link2++)
	if (link1 != link2){
	  count++;
	  corner=ishift(color,link1,x);
	  corner=ishift(corner,link2,y);
	  forvector{
	    mtemp[0][iv]=1.;
	    mtemp[1][iv]=1.;
	    mtemp[2][iv]=1.;
	    mtemp[3][iv]=1.;
	  }
	  for(i=0;i<x;i++){
	    getlinks(mtemp[4],u,ishift(color,link1,i),link1);
	    vprod(mtemp[0],mtemp[4],mtemp[0]);
	    getconjugate(mtemp[4],u,ishift(corner,link1,-i-1),link1);
	    vprod(mtemp[2],mtemp[4],mtemp[2]);
	  }
	  for(i=0;i<y;i++){
	    getlinks(mtemp[4],u,ishift(corner,link2,i-y),link2);
	    vprod(mtemp[1],mtemp[4],mtemp[1]);
	    getconjugate(mtemp[4],u,ishift(color,link2,y-i-1),link2);
	    vprod(mtemp[3],mtemp[4],mtemp[3]);
	  }
	  vprod(mtemp[0],mtemp[1],mtemp[0]);
	  vprod(mtemp[0],mtemp[2],mtemp[0]);
	  vtprod(mtemp[0],mtemp[3],sold);
	  forvector
	    result+=sold[iv];
	}
  result=result/(GROUP*vectorlength*count);
  printf(" %d by %d loop = %g\n",x,y,result);
  return result;
}

  /* utilities for implementing periodic boundaries */

int siteindex(int *x){
  /* gives a unique index to site located at x[DIM] */
  int i,result=0;
  for (i=0;i<DIM;i++)
    result+=shift[i]*x[i];
  return result;
}

void split(int *x, int s){
  /* splits a site index into coordinates */
  /* assume s in valid range 0<=s<nsites */
  /* I think this is faster than using mods, but this should be tested */
  int i;
  if (s<0 || s>=nsites) cleanup("bad split");
  for(i=DIM-1;i>0;i--){
    x[i]=0;
    while (s>=shift[i]){
      s-=shift[i];
      x[i]++;
    }
  }
  x[0]=s;
  return;
}

int vshift(int n, int *x){
  /* shifts site n by vector x */
  int i,y[DIM];
  split(y,n);
  for(i=0;i<DIM;i++){
    if (x[i]){
      y[i]+=x[i];
      while (y[i]>=shape[i])
	y[i]-=shape[i];
      while (y[i]<0)
	y[i]+=shape[i];
    }
  }
  return siteindex(y);
}

int ishift( int n, int dir, int dist){
  /* returns index of a site shifted dist in direction dir */
  int i,x[DIM];
  for(i=0;i<DIM;i++)
    x[i]=0;
  x[dir]=dist;
  return vshift(n,x);
}

  /* matrix manipulation routines */


void matrix::printmatrix(){
  int i,j;
  for (i=0;i<GROUP;i++){
    printf("\n");
    for (j=0;j<GROUP;j++){
      printf(" (%g, %g)   ",real[i][j],imag[i][j]);
    }
  }
  printf("\n");
}

inline matrix operator* (matrix& lhs, matrix& rhs){
  int i,j,k;
  matrix result;
  result=0.;
  for(i=0;i<GROUP;i++)
    for(j=0;j<GROUP;j++)
      for(k=0;k<GROUP;k++){
	result.real[i][j]+=(lhs.real[i][k]*rhs.real[k][j]
			    -lhs.imag[i][k]*rhs.imag[k][j]);
	result.imag[i][j]+=(lhs.real[i][k]*rhs.imag[k][j]
			    +lhs.imag[i][k]*rhs.real[k][j]);
      }
  return result;
}

inline matrix operator+ (matrix& lhs, matrix& rhs){
  int i,j;
  matrix result;
  result=0.;
  for(i=0;i<GROUP;i++)
    for(j=0;j<GROUP;j++){
      result.real[i][j]+=(lhs.real[i][j]+rhs.real[i][j]);
      result.imag[i][j]+=(lhs.imag[i][j]+rhs.imag[i][j]);
    }
  return result;
}

inline matrix operator- (matrix& lhs, matrix& rhs){
  int i,j;
  matrix result;
  result=0.;
  for(i=0;i<GROUP;i++)
    for(j=0;j<GROUP;j++){
      result.real[i][j]+=(lhs.real[i][j]-rhs.real[i][j]);
      result.imag[i][j]+=(lhs.imag[i][j]-rhs.imag[i][j]);
    }
  return result;
}

matrix& matrix::operator= (double x) {
  for(int i=0;i<GROUP;i++)
    for(int j=0;j<GROUP;j++) {
      real[i][j]= ((i==j) ? x : 0.);
      imag[i][j]=0.;
    }
  return *this;
}

matrix& matrix::operator*= (double x) {
  for(int i=0;i<GROUP;i++)
    for(int j=0;j<GROUP;j++) {
      real[i][j]*=x;
      imag[i][j]*=x;
    }
  return *this;
} 

matrix conjugate(matrix g) {
  matrix result;
  for(int i=0;i<GROUP;i++)for(int j=0;j<GROUP;j++) {
      result.real[i][j]= g.real[j][i];
      result.imag[i][j]= -g.imag[j][i];
    }
  return result;
}

matrix& matrix::project() {
  /* projects a matrix onto the group SU(GROUP) */
  int i,j,k,nmax;
  nmax=GROUP-(GROUP<4); /* 2 and 3 are treated specially */
  /* loop over rows */
  for (i=0;i<nmax;i++) {
    /* normalize i'th row */
    double temp=(*this).real[i][0]*(*this).real[i][0]
      +(*this).imag[i][0]*(*this).imag[i][0];
    for (j=1;j<GROUP;j++)
      temp+=(*this).real[i][j]*(*this).real[i][j]
	+(*this).imag[i][j]*(*this).imag[i][j];
    temp=1/sqrt(temp);   
    for (j=0;j<GROUP;j++) {
      (*this).real[i][j]*=temp;
      (*this).imag[i][j]*=temp;
    }
    /* orthogonalize remaining rows */
    double adotbr,adotbi;
    for (k=i+1;k<nmax;k++) {
      adotbr=(*this).real[i][0]*(*this).real[k][0]
	+(*this).imag[i][0]*(*this).imag[k][0];
      adotbi= (*this).real[i][0]*(*this).imag[k][0]
	-(*this).imag[i][0]*(*this).real[k][0];
      for (j=1;j<GROUP;j++) {
	adotbr+=(*this).real[i][j]*(*this).real[k][j]
	  +(*this).imag[i][j]*(*this).imag[k][j];
	adotbi+= (*this).real[i][j]*(*this).imag[k][j]
	  -(*this).imag[i][j]*(*this).real[k][j];
      }
      for (j=0;j<GROUP;j++) {
	(*this).real[k][j]-=adotbr*(*this).real[i][j]
	  -adotbi*(*this).imag[i][j];
	(*this).imag[k][j]-= adotbr*(*this).imag[i][j]
	  +adotbi*(*this).real[i][j];
      } 
    } /* end of k loop */
  } /* end of i loop */
    /* remove determinant, treating group=2 or 3 as special cases */
  switch (GROUP) {
  case 3:
    thirdrow(*this);
    break; 
  case 2: /* for su(2) */
    (*this).real[1][0]=-(*this).real[0][1];
    (*this).real[1][1]= (*this).real[0][0];
    (*this).imag[1][0]= (*this).imag[0][1];
    (*this).imag[1][1]=-(*this).imag[0][0];
    break;  
  default: /* remove the determinant from the first row */
    double x,y,w;
    (*this).determinant(x,y);
    for (i=0;i<GROUP;i++) {
      w=(*this).real[0][i]*x
	+(*this).imag[0][i]*y;
      (*this).imag[0][i]=(*this).imag[0][i]*x
	-(*this).real[0][i]*y;
      (*this).real[0][i]=w;
    }
  } /* end switch */
  return *this;
}

void matrix::determinant(double& detr, double& deti) {
  /* subroutine to calculate matrix determinant */
  /* could perhaps improve later by doing group=2 and 3 by hand,
     but as of now these cases don't call this routine anyway */
  int i,j,k,im,km;
  double temp;
  matrix copy=*this;
  im=GROUP-1;
  for (i=0;i<im;i++) {
    km=i+1;
    /* find magnitude of i'th diagonal element */
    temp=1./(copy.real[i][i]*copy.real[i][i]
	     +copy.imag[i][i]*copy.imag[i][i]);
    for (k=km;k<GROUP;k++) {
      /* subtract part of row i from row k to make [k][i] element vanish */
      /* inner product of the rows */
      detr=(copy.real[k][i]*copy.real[i][i]
	    +copy.imag[k][i]*copy.imag[i][i])*temp;
      deti=(copy.imag[k][i]*copy.real[i][i]
	    -copy.real[k][i]*copy.imag[i][i])*temp;
      for (j=km;j<GROUP;j++) {
	copy.real[k][j]+= -detr*copy.real[i][j]
	  +deti*copy.imag[i][j];
	copy.imag[k][j]+=  -detr*copy.imag[i][j]
	  -deti*copy.real[i][j];
      }
    }  
  }  
  /* multiply diagonal elements */
  detr=copy.real[0][0];
  deti=copy.imag[0][0];
  for (i=1;i<GROUP;i++) {
    temp=detr*copy.real[i][i]-deti*copy.imag[i][i];
    deti=deti*copy.real[i][i]+detr*copy.imag[i][i];
    detr=temp;
  }
  return;
}
