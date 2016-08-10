/*
A function that finds all pairs with a given separation with an operation on the pairs 

Required Input
    *op_samples  :: double* array (n,L,...,L) :: where (L, ..., L) coresponds to the lattice at a set time point
    *sep_map     :: double* array (n) :: map separations to op_samples
    *ans         :: double* ::  the outputted result
    *n           :: int* :: lenght of time sample - as above
    *count       :: int* :: count of matching pairs
    tol          :: double :: tolerance - also used as binning method
    sep          :: double :: the separation to match pairs for

Returns / Modified quantities
    ans, count
*/

void autocorrelation (double* ans,        // return value
    double* op_samples,                  // M-length an array of hyper-cubes of length L
    int* front, int* back, double* mean,
    int sites
    ){
    // Don't know what to do with this section as this will be a nested
    // series of d for loops  for d dimensions
    ans += np.mean((op_samples[front] - mean)*(op_samples[back] - mean));
}

void acorrMapped_noDup (
    double* op_samples, // M-length an array of hyper-cubes of length L
    double* sep_map,    // 1 dimensional array of length, M
    double* ans, int* count, int n, double tol, double sep) {
    
    int front = 1;   // front pointer
    int back  = 0;   // back pointer
    *ans   = 0.0; // the answer
    *count = 0;         // counter for averaging
    
    while (front < n) { // keep going until exhausted sep_map
        diff = sep_map[front] - sep_map[back] - sep;
        if abs(diff) < tol{ // if equal subject to tol: pair found
            count++;
            autocorrelation(&ans, &op_samples, &front, &back, &mean)
            back ++;
            front++;
        }
        else if (diff > sep){ back ++; }
        else { front ++; }
    }
}