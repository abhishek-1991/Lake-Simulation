Group Details:
**ajain28 Abhash Jain
**asrivas3 Abhishek Kumar

Project Title: Implementation of Lake Simulation using 9 Pt. Stencil using CUDA and CUDA/MPI.

Introduction: In this project a Lake Simulation is provided using a 5 point stencil. Our Task is to implement the Simulation using a 9 Point Stencil
by performing calculations sequentially on CPU, on GPU using Cuda and on distributed nodes with GPU using MPI and Cuda.

PREQUISTIES: Compilers capable of compiling MPI programs and Cuda Programs and 4 nodes consisting of Nvidia GPUs.

V1:
------
Solution: This part is implemented for sequential compuations (as asked in question). The only change was made in the formula for calculating the image points.
Hence images were generated for 9 point stencils.

Inference:
1. The ripples in images of 9 point stencil were expanding at greater rate than 5 point stencil.
2. The images generated by 9 point stencil had circular ripples while in 5 point stencil they were more square in shape with slightly curved edges.

V2:
------
Solution: This part is implemented by performing calculations on GPU using CUDA. Each pixel was calculated on a different thread for each iteration of "dt".
After that the results were collated from each GPU on the host and the data points were returned for heatmap generation.

Results:
Following Results were obtained for 5 pebbles, end time 1.0 seconds and 8 number of threads:
			CPU			GPU
16		0.008403	0.521273
32		0.004145	0.315599
64		0.03541		0.315957
128		0.286269	0.38764
256		2.415816	0.416013
512		19.813948	0.525251
1024	180.184736	1.416748

Inferences:
For Small grid size the execution time for CPU is faster than GPU but as the grid size increases the the execution time for CPU increases exponentially while
the execution time for GPU is increasing very slowly. Hence, for large grids the CPU is taking very long time to execute while GPU is taking significantly less
time than CPU.

V4:
------
Solution: This part is implemented by dividing the grid between four nodes and passing the border values to each other using MPI and then on each node the calculations
are performed on GPU using Cuda for faster calculations. After that the resulting grid values are collated from each GPU and dat file is generated for each node separately.

Inference:
1. Integrating CUDA and MPI involves more sophisticated code because appropriate boundary values needs to be communicated among nodes and these values must also be moved to
the device storage for GPU to perform calculations. Hence, finding the appropriate split between the data was major challenge we faced. We fixed them by devicing a formula to
find the boundary indexes that needs to communicated to the adjacent nodes.
2. If the Data is Large then due to communication delays the Calculation takes more time than the code when not parallelized on multiple nodes.
3. If the grid is perfect square of number threads then the performance is much better.
4. This program does not perform well when the grid size is very large. This might be caused due to network overheads and small size of GPU Device memory.

How to Run the program:
Get the number of processor using srun Command from the arc cluster.
E.g. : srun -N4 -n 4 -p opteron --pty /bin/bash
1. Copy the below file to the working directlory

2. Compile the code using Make file
V1 & V2:
---------
$ make -f p3.Makefile lake
V4:
---------
$ make -f p3.Makefile lake-mpi

3. To run the program 
V1 & V2:
--------
$ ./lake <number of grid points> <number of pebbles> <end time> <number of threads>
e.g:  ./lake 128 5 1.0 8
V4:
--------
$ prun ./lake <number of grid points> <number of pebbles> <end time> <number of threads>
e.g:  prun ./lake 128 5 1.0 8

Files submitted:
1. lake.cu
2. lakegpu.cu
3. lake_mpi.cu
4. lakegpu_mpi.cu
5. p3.Makefile
6. lake5pt.png
7. lake9pt.png
8. p2.README
---------------------------------End Of Document--------------------------------------
