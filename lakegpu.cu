/*******************************************************
 * Author:
 * asrivas3 Abhishek Kumar Srivastava
 * ajain28 Abhash Jain
 * CSC 548 Assignment#2 Problem#3
 ******************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define VSQR 0.1
#define TSCALE 1.0
#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

int tpdt(double *t, double dt, double tf);


/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

//function for calculating exponent
__device__ double f1(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

//kernel function for gpu
__global__ void evolve_gpu(double *un,double *uc, double *u0, double *pebbles, int *n, double *h, double *dt,double *t){
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int i = idx/(*n);
	int j= idx%(*n);
	
	if(i == 0 || i == ((*n)-1) || j == 0 || j == ((*n) -1) ){
		un[idx]=0;
	}
	else {
		un[idx] = 2*uc[idx] - u0[idx] + VSQR *((*dt) * (*dt)) *((uc[idx-1] + uc[idx+1] + 
                    uc[idx + (*n)] + uc[idx - (*n)] +0.25*(uc[idx - (*n) + 1] + uc[idx + (*n) -1 ] 
					+ uc[idx - (*n) - 1]+uc[idx + (*n) + 1])  - 5 * uc[idx])/((*h) * (*h)) + f1(pebbles[idx],(*t)));
	}
}


void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
        
	/* HW2: Define your local variables here */
	double* u_d;
	double* u0_d;
	double* u1_d;
	double* peb_d;
	int *n_d;
	double *h_d;
	double *dt_d;
	double *t_d;
	double t=0.,dt=h/2.;
	double *temp;

    /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
	cudaMalloc((void **)&u_d,n*n*sizeof(double));
	cudaMalloc((void **)&u0_d,n*n*sizeof(double));
	cudaMalloc((void **)&u1_d,n*n*sizeof(double));
	cudaMalloc((void **)&peb_d,n*n*sizeof(double));
	cudaMalloc((void **)&n_d,sizeof(int));
	cudaMalloc((void **)&h_d,sizeof(double));
	cudaMalloc((void **)&dt_d,sizeof(double));
	cudaMalloc((void **)&t_d,sizeof(double));
	
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));
	
	cudaMemcpy(u0_d,u0,sizeof(double)*n*n,cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d,u1,sizeof(double)*n*n,cudaMemcpyHostToDevice);
	cudaMemcpy(peb_d,pebbles,sizeof(double)*n*n,cudaMemcpyHostToDevice);
	cudaMemcpy(n_d,&n,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(h_d,&h,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dt_d,&dt,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(t_d,&t,sizeof(double),cudaMemcpyHostToDevice);
	
	int block = n/nthreads;
	block*=block;
	int threads = nthreads*nthreads;
	
	/* HW2: Add main lake simulation loop here */
	while(1){
		evolve_gpu<<<block,threads>>>(u_d,u1_d,u0_d,peb_d,n_d,h_d,dt_d,t_d);
		temp = u1_d;
		u1_d = u_d;
		u_d = u0_d;
		u0_d = temp;
		if(!tpdt(&t,dt,end_time)) break;
		cudaMemcpy(t_d,&t,sizeof(double),cudaMemcpyHostToDevice);
	}
	cudaMemcpy(u,u1_d,sizeof(double)*n*n,cudaMemcpyDeviceToHost);
	
    /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
	cudaFree(u_d);
	cudaFree(u0_d);
	cudaFree(u1_d);
	cudaFree(peb_d);
	cudaFree(n_d);
	cudaFree(h_d);
	cudaFree(dt_d);
	cudaFree(t_d);
	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
