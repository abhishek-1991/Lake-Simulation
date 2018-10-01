/*
* HW2 - Assignment 2 - Question 3 V4
* Description : Lake ripple simulation using CUDA and MPI
* Author : 
* ajain28  Abhash Jain
* asrivas3 Abhishek kumar srivastava
*
*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

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

__device__ double f1(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}


__global__ void evolve_gpu(double *un,double *uc, double *u0, double *pebbles, int *n, double *h, double *dt,double *t,int *m_size){
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if(idx<(*m_size)){
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
	}//if size check condn.
}

/*int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}*/

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads,int my_rank,int m_size)
{
	cudaEvent_t kstart, kstop;
	float ktime;
	/* HW2: Define your local variables here */
	
	double *u_d,*u0_d,*u1_d,*peb_d,*u1_host,*u0_host,*un_host;
	int *n_d,*m_size_d;
	double *h_d;
	double *dt_d;
	double *t_d;
	double t=0.,dt=h/2.;
	int tag =50;
	MPI_Status status;
	

    /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
	u1_host = (double*)malloc(sizeof(double)*m_size);
    u0_host = (double*)malloc(sizeof(double)*m_size);
    un_host = (double*)malloc(sizeof(double)*m_size);
	memset(u0_host,0,sizeof(double)*m_size);
	cudaMalloc((void **)&u_d,m_size*sizeof(double));
	cudaMalloc((void **)&u0_d,m_size*sizeof(double));
	cudaMalloc((void **)&u1_d,m_size*sizeof(double));
	cudaMalloc((void **)&peb_d,m_size*sizeof(double));
	cudaMalloc((void **)&n_d,sizeof(int));
	cudaMalloc((void **)&m_size_d,sizeof(int));
	cudaMalloc((void **)&h_d,sizeof(double));
	cudaMalloc((void **)&dt_d,sizeof(double));
	cudaMalloc((void **)&t_d,sizeof(double));
	
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	//copy memory to store value on kernal
	cudaMemcpy(u_d,u0_host,sizeof(double)*m_size,cudaMemcpyHostToDevice);
	cudaMemcpy(u0_d,u0,sizeof(double)*m_size,cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d,u1,sizeof(double)*m_size,cudaMemcpyHostToDevice);
	cudaMemcpy(peb_d,pebbles,sizeof(double)*m_size,cudaMemcpyHostToDevice);
	cudaMemcpy(n_d,&n,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(m_size_d,&m_size,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(h_d,&h,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dt_d,&dt,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(t_d,&t,sizeof(double),cudaMemcpyHostToDevice);
	
	int block = m_size/(nthreads*nthreads);
	//block*=block;
	int threads = nthreads*nthreads;
	
	/* HW2: Add main lake simulation loop here */
	while(1){
		//call the kernal
		evolve_gpu<<<block,threads>>>(u_d,u1_d,u0_d,peb_d,n_d,h_d,dt_d,t_d,m_size_d);

		//get the latest value from the node and send boundry values to nearby nodes
		cudaMemcpy(un_host,u_d,sizeof(double)*m_size,cudaMemcpyDeviceToHost);
		cudaMemcpy(u1_host,u1_d,sizeof(double)*m_size,cudaMemcpyDeviceToHost);
		//rank 0 send and recv from node 1 
		// rank 1 updtes itvalues from node 0 and 2, send to 0,2
		//rank 3 nodes only send update value from 2
		if(my_rank==0){
        	MPI_Send(u1_host+(m_size-(2*n)),n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
	        MPI_Send(un_host+(m_size-(2*n)),n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
    	    MPI_Recv(u1_host+(m_size-n),n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
        	MPI_Recv(un_host+(m_size-n),n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
    	} else if(my_rank==1){
        	MPI_Recv(u1_host,n,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
	        MPI_Recv(un_host,n,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
    	    MPI_Send(u1_host+n,n,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
        	MPI_Send(un_host+n,n,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
	        MPI_Send(u1_host+(m_size-(2*n)),n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD);
    	    MPI_Send(un_host+(m_size-(2*n)),n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD);
			MPI_Recv(u1_host+(m_size-n),n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD,&status);
        	MPI_Recv(un_host+(m_size-n),n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD,&status);
    	} else if(my_rank==2){
        	MPI_Recv(u1_host,n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
	        MPI_Recv(un_host,n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
    	    MPI_Send(u1_host+n,n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
        	MPI_Send(un_host+n,n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
	        MPI_Send(u1_host+(m_size-(2*n)),n,MPI_DOUBLE,3,tag,MPI_COMM_WORLD);
    	    MPI_Send(un_host+(m_size-(2*n)),n,MPI_DOUBLE,3,tag,MPI_COMM_WORLD);
        	MPI_Recv(u1_host+(m_size-n),n,MPI_DOUBLE,3,tag,MPI_COMM_WORLD,&status);
	        MPI_Recv(un_host+(m_size-n),n,MPI_DOUBLE,3,tag,MPI_COMM_WORLD,&status);
    	}else{
        	MPI_Recv(u1_host,n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD,&status);
	        MPI_Recv(un_host,n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD,&status);
    	    MPI_Send(u1_host+n,n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD);
        	MPI_Send(un_host+n,n,MPI_DOUBLE,2,tag,MPI_COMM_WORLD);
    	}
		//copy back the value to device memory after exchange
		cudaMemcpy(u0_d,u1_host,sizeof(double)*m_size,cudaMemcpyHostToDevice);
		cudaMemcpy(u1_d,un_host,sizeof(double)*m_size,cudaMemcpyHostToDevice);
		if(!tpdt(&t,dt,end_time)) break;
		cudaMemcpy(t_d,&t,sizeof(double),cudaMemcpyHostToDevice);
	}
	//get the final result
	cudaMemcpy(u,u_d,sizeof(double)*m_size,cudaMemcpyDeviceToHost);
	
    /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
	//free memory
	cudaFree(u_d);
	cudaFree(u0_d);
	cudaFree(u1_d);
	cudaFree(peb_d);
	cudaFree(n_d);
	cudaFree(h_d);
	cudaFree(dt_d);
	cudaFree(t_d);
	free(u1_host);
	free(u0_host);
	free(un_host);
	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
