# GPU Accelerated Computing: CUDA and OpenCL in C/C++

This repository presents my projects in GPU programming, leveraging both NVIDIA's CUDA platform and the open-standard OpenCL framework. The goal is to demonstrate the power of highly parallel GPU architectures for accelerating computationally intensive tasks, with a focus on kernel optimization and efficient data transfer.

## Key Learning Outcomes & Skills:

* **CUDA C/C++:** Writing `device kernels`, `host-device memory management (cudaMalloc, cudaMemcpy)`, `thread hierarchy (grids, blocks, threads)`, `shared memory optimization (Challenge)`, `CUDA events for timing`.
* **OpenCL C:** Developing `OpenCL kernels`, `host-side API (platforms, devices, contexts, command queues, buffers)`, `global/local work-items`, `cross-vendor GPU programming`.
* **Performance Optimization:** Analyzing execution times on GPUs versus CPUs, identifying bottlenecks, and optimizing memory access patterns.
* **Numerical Computing:** Applying GPU acceleration to common engineering problems (Matrix Multiplication, Heat Diffusion/Laplace Solver, Numerical Integration).
* **Hardware Awareness:** Justifying workgroup sizes based on GPU architecture (e.g., streaming multiprocessors, shared memory).
* **Correctness Verification:** Comparing GPU results against CPU versions for accuracy.
* **Heterogeneous Computing (Extra Challenge):** Utilizing multiple devices (CPU/GPU) in parallel with OpenCL/multi-threading.

## Projects:

### 1. OpenCL Vector Addition
* **Description:** A foundational OpenCL program demonstrating basic vector addition, showing the host-kernel interaction and data transfer.
* **Skills Showcased:** `OpenCL environment setup`, `kernel creation`, `buffer management`, `kernel argument setting`, `NDRange execution`.
* **Source:** `CS435 Lab 09 Manual - GPU Programming with OpenCL.pdf` (Lab Task 1)
* **Details:** `opencl_vector_add.cpp` (host code) and `vector_add_kernel.cl` (kernel code).

### 2. OpenCL Multiple Vector Addition
* **Description:** Extends the basic vector addition to sum five input arrays into an output array using a single OpenCL kernel.
* **Skills Showcased:** `Extending kernel functionality`, `handling multiple input buffers`, `efficient data processing on GPU`.
* **Source:** `CS435 Lab 09 Manual - GPU Programming with OpenCL.pdf` (Lab Task 2)
* **Details:** `opencl_multi_vector_add.cpp` (host code) and `multi_vector_add_kernel.cl` (kernel code).

### 3. OpenCL vs. OpenMP Dot Product Comparison
* **Description:** Compares the performance of dot product calculation using OpenCL on a GPU and OpenMP on a CPU for large vectors.
* **Skills Showcased:** `OpenCL for reduction`, `local memory (shared memory in CUDA terms) optimization`, `OpenMP reduction`, `performance benchmarking`.
* **Source:** `CS435 Lab 09 Manual - GPU Programming with OpenCL.pdf` (Lab Task 3)
* **Details:** `opencl_dot_product.cpp`, `dot_product_kernel.cl`, `openmp_dot_product.cpp`.

### 4. OpenCL vs. OpenMP Scalar Matrix Multiplication
* **Description:** Compares OpenCL (GPU) and OpenMP (CPU) implementations for scalar multiplication of a large matrix.
* **Skills Showcased:** `2D global work-items in OpenCL`, `OpenMP parallel for`, `matrix element-wise operations`.
* **Source:** `CS435 Lab 09 Manual - GPU Programming with OpenCL.pdf` (Lab Task 4)
* **Details:** `opencl_scalar_matmul.cpp`, `scalar_matmul_kernel.cl`, `openmp_scalar_matmul.cpp`.

### 5. OpenCL vs. OpenMP Matrix Multiplication (Naive)
* **Description:** Benchmarks naive dense matrix multiplication (C = A * B) using OpenCL (GPU) and OpenMP (CPU) for varying matrix sizes, ensuring correctness.
* **Skills Showcased:** `GPU matrix multiplication kernel design`, `efficient memory access patterns for matrix multiplication`, `OpenCL 2D NDRange`, `performance scaling`, `correctness verification`.
* **Source:** `CS435 Lab 09 Manual - GPU Programming with OpenCL.pdf` (Lab Task 5) and `PDP-Spring2025-Assignment-04-23042025 (1).pdf` (Question 1)
* **Details:** `opencl_matrix_mul_naive.cpp` (host), `matrix_mul_naive_kernel.cl` (kernel), and `openmp_matrix_mul_naive.cpp`.
* **Results:** Includes detailed time comparisons and correctness checks (sum of differences should be zero).
* **Justification of Workgroup Size:** Discussion on `local_size` selection for optimal performance based on typical GPU architecture properties (e.g., multiples of warp/wavefront size, shared memory limits).

### 6. OpenCL 1D Convolution
* **Description:** Implements 1D convolution of an input vector with a filter vector using OpenCL, showcasing host-side setup and kernel execution.
* **Skills Showcased:** `OpenCL for signal processing (convolution)`, `1D global work-items`, `kernel design for sliding window operations`.
* **Source:** `CS435 Lab 10 Manual - Solving Engineering Problems using OpenCL.pdf` (Lab Task 1)
* **Details:** `opencl_1d_convolution.cpp` (host) and `conv1D_kernel.cl` (kernel).
* **Results:** Time measurements and potentially a graph plotting "Scale vs Time Taken".

### 7. OpenCL 2D Convolution
* **Description:** Extends the 1D convolution to a 2D version, applying a filter to a large 2D input matrix using OpenCL.
* **Skills Showcased:** `2D convolution algorithm on GPU`, `OpenCL 2D NDRange for image processing`, `efficient matrix indexing in kernels`.
* **Source:** `CS435 Lab 10 Manual - Solving Engineering Problems using OpenCL.pdf` (Lab Task 2)
* **Details:** `opencl_2d_convolution.cpp` (host) and `conv2D_kernel.cl` (kernel).
* **Results:** Performance data for various matrix sizes, visualized with a "Scale vs Time Taken for 2D Convolution" graph.

### 8. OpenCL Numerical Integration
* **Description:** Parallelizes numerical integration of several mathematical functions using OpenCL, applying the rectangle rule on the GPU.
* **Skills Showcased:** `Numerical integration on GPU`, `OpenCL function kernels`, `broadcasting parameters to kernel`, `high precision floating-point arithmetic`.
* **Source:** `CS435 Lab 10 Manual - Solving Engineering Problems using OpenCL.pdf` (Lab Task 4)
* **Details:** `opencl_numerical_integration.cpp` (host) and `integral_kernel.cl` (kernel).

### 9. CUDA 2D Heat Diffusion / Laplace Solver
* **Description:** Simulates 2D heat diffusion using the finite difference method, highly accelerated on an NVIDIA GPU using CUDA. This also implicitly solves Laplace's equation for steady-state heat distribution.
* **Skills Showcased:** `CUDA kernel development`, `2D thread block and grid configuration`, `ping-pong buffering` for iterative updates, `host-device memory management`, `CUDA event timing`, `boundary clamping (finite difference stencil)`, `handling toolchain compatibility issues (gencode flags)`.
* **Source:** `CS435 Lab 11 Manual - Solving Engineering Problems using CUDA.pdf` (Problem Statement) and `PDP-Spring2025-Assignment-04-23042025 (1).pdf` (Question 2)
* **Details:** `cuda_heat_diffusion_laplace_solver.cu`.
    * **Problem:** Solve $\frac{\partial T}{\partial t} = \alpha (\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2})$ using finite differences.
    * **Boundary Conditions:** Implicitly handled by clamping for heat diffusion. If explicitly for Laplace, mention fixed `top=5V, bottom=-5V, left=right=0V`.
    * **Optimization:** Discussion on the choice of `threadsPerBlock` (`dim3 threadsPerBlock (16, 16);`) and `numBlocks` for optimal GPU utilization.
    * **Correctness:** Comparison against single-threaded CPU version.
    * **Challenge (Future/Mention):** Opportunity to implement a shared/local memory optimized version.
* **Results:** Performance screenshots (e.g., "Total Simulation Time: 101.506020 ms" for N=1000, TIME_STEPS=1000) and correctness output.

### 10. OpenCL 2D Laplace Solver 
* **Description:** Implements a 2D Laplace solver using OpenCL, comparable to the CUDA version, with specific boundary conditions.
* **Skills Showcased:** `OpenCL kernel for Laplace solver`, `device interaction`, `workgroup size justification`, `correctness verification`.
* **Source:** `PDP-Spring2025-Assignment-04-23042025 (1).pdf` (Question 2)
* **Details:** `opencl_laplace_solver.cpp` (host) and `laplace_kernel.cl` (kernel). Implement using `top=5V, bottom=-5V, left=right=0V` boundary conditions.
* **Justification of Workgroup Size:** Explanation of optimal workgroup dimensions for OpenCL on a given device.
* **Challenge (Future/Mention):** Opportunity to implement a shared/local memory optimized version.

### 11. CUDA Matrix Multiplication (Naive)
* **Description:** Implements naive matrix multiplication using CUDA, with performance and correctness comparison against CPU versions.
* **Skills Showcased:** `CUDA kernel for matrix multiplication`, `host-device memory management`, `thread block/grid design`, `performance benchmarking`, `correctness verification`.
* **Source:** `PDP-Spring2025-Assignment-04-23042025 (1).pdf` (Question 1)
* **Details:** `cuda_matrix_mul_naive.cu`.
* **Justification of Workgroup Size:** Discussion on `blockDim` and `gridDim` selection.

### (Optional) Heterogeneous Computing 
* **Description:** Design for a heterogeneous computing solution using OpenCL across multiple devices (e.g., CPU/GPU) for a parallel problem.
* **Skills Showcased:** `OpenCL multi-device management`, `task distribution across heterogeneous hardware`, `multi-threading for parallel device execution`.
* **Source:** `PDP-Spring2025-Assignment-04-23042025 (1).pdf` (Extra Challenge)
* **Details:** (This would be a conceptual write-up or a small proof-of-concept if implemented). Explain the strategy for dividing work and managing synchronization across different device types.
