%%writefile q2_laplace_solver_cuda.cu
#include <iostream>
#include <vector>
#include <cmath>   // For fabs()
#include <iomanip> // For std::fixed, std::setprecision
#include <omp.h>   // For OpenMP functions and pragmas
#include <cuda_runtime.h>
#include <cstdlib> // For malloc, free, exit
#include <ctime>   // For srand, time (if needed, though not for Laplace init)

// Helper function for CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Grid dimensions and parameters for Question 2
const int GRID_DIM_Q2 = 512;      // Dimension of the square grid (e.g., 512x512)
const int MAX_ITERATIONS_Q2 = 1000; // Number of iterations for the solver

// Boundary conditions for Question 2
const float TOP_VOLTAGE_Q2 = 5.0f;
const float BOTTOM_VOLTAGE_Q2 = -5.0f;
const float LEFT_VOLTAGE_Q2 = 0.0f;
const float RIGHT_VOLTAGE_Q2 = 0.0f;
const float INITIAL_GUESS_Q2 = 0.0f; // Initial guess for interior points

const float ERROR_THRESHOLD_Q2 = 1e-3f; // Threshold for floating point comparison

// --- Initialize Grid with Boundary Conditions and Initial Guess ---
void initialize_grid_laplace_q2(float* grid, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0) { // Top boundary
                grid[i * N + j] = TOP_VOLTAGE_Q2;
            } else if (i == N - 1) { // Bottom boundary
                grid[i * N + j] = BOTTOM_VOLTAGE_Q2;
            } else if (j == 0) { // Left boundary
                grid[i * N + j] = LEFT_VOLTAGE_Q2;
            } else if (j == N - 1) { // Right boundary
                grid[i * N + j] = RIGHT_VOLTAGE_Q2;
            } else { // Interior points
                grid[i * N + j] = INITIAL_GUESS_Q2;
            }
        }
    }
}

// --- CPU Single-Threaded Laplace Solver (Jacobi Iteration) ---
void laplace_solver_cpu_single_q2(float* current_grid, float* next_grid, int N, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Update interior points
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next_grid[i * N + j] = 0.25f * (
                    current_grid[(i - 1) * N + j] +  // Top neighbor
                    current_grid[(i + 1) * N + j] +  // Bottom neighbor
                    current_grid[i * N + (j - 1)] +  // Left neighbor
                    current_grid[i * N + (j + 1)]    // Right neighbor
                );
            }
        }
        // Copy next_grid back to current_grid for the next iteration
        // Only interior points need to be copied as boundaries are fixed.
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                current_grid[i * N + j] = next_grid[i * N + j];
            }
        }
    }
}

// --- CPU OMP Laplace Solver (Jacobi Iteration) ---
void laplace_solver_cpu_omp_q2(float* current_grid, float* next_grid, int N, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Update interior points in parallel
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next_grid[i * N + j] = 0.25f * (
                    current_grid[(i - 1) * N + j] +
                    current_grid[(i + 1) * N + j] +
                    current_grid[i * N + (j - 1)] +
                    current_grid[i * N + (j + 1)]
                );
            }
        }
        // Copy next_grid back to current_grid for the next iteration
        // This copy can also be parallelized.
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                current_grid[i * N + j] = next_grid[i * N + j];
            }
        }
    }
}

// --- CUDA Kernel for Laplace Solver (Jacobi Iteration) ---
__global__ void laplace_kernel_cuda_q2(const float* current_grid, float* next_grid, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < N - 1 && col > 0 && col < N - 1) { // Only update interior points
        next_grid[row * N + col] = 0.25f * (
            current_grid[(row - 1) * N + col] +
            current_grid[(row + 1) * N + col] +
            current_grid[row * N + (col - 1)] +
            current_grid[row * N + (col + 1)]
        );
    }
}

// --- Correctness Check ---
bool verify_laplace_results_q2(const float* ref_grid, const float* other_grid, int N, float threshold, float& max_abs_diff, float& avg_abs_diff) {
    max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    int interior_points_count = 0;

    for (int i = 1; i < N - 1; ++i) { // Only compare interior points
        for (int j = 1; j < N - 1; ++j) {
            float diff = fabs(ref_grid[i * N + j] - other_grid[i * N + j]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            sum_abs_diff += diff;
            interior_points_count++;
        }
    }
    avg_abs_diff = (interior_points_count > 0) ? static_cast<float>(sum_abs_diff / interior_points_count) : 0.0f;
    
    return max_abs_diff <= threshold;
}

int main() {
    std::cout << std::fixed << std::setprecision(6); 

    size_t grid_elements = (size_t)GRID_DIM_Q2 * GRID_DIM_Q2;
    size_t grid_size_bytes = grid_elements * sizeof(float);

    // Host memory
    float* h_grid_cpu_single = (float*)malloc(grid_size_bytes);
    float* h_grid_cpu_single_next = (float*)malloc(grid_size_bytes);
    float* h_grid_cpu_omp = (float*)malloc(grid_size_bytes);
    float* h_grid_cpu_omp_next = (float*)malloc(grid_size_bytes);
    float* h_grid_gpu_result = (float*)malloc(grid_size_bytes);

    if (!h_grid_cpu_single || !h_grid_cpu_single_next || !h_grid_cpu_omp || !h_grid_cpu_omp_next || !h_grid_gpu_result) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        // Basic cleanup
        free(h_grid_cpu_single); free(h_grid_cpu_single_next); free(h_grid_cpu_omp); free(h_grid_cpu_omp_next); free(h_grid_gpu_result);
        return 1;
    }
    
    double start_time, end_time;
    float max_diff_val, avg_diff_val;

    // --- CPU Single-Threaded Execution ---
    initialize_grid_laplace_q2(h_grid_cpu_single, GRID_DIM_Q2);
    initialize_grid_laplace_q2(h_grid_cpu_single_next, GRID_DIM_Q2); // Also init boundaries for next buffer

    start_time = omp_get_wtime();
    laplace_solver_cpu_single_q2(h_grid_cpu_single, h_grid_cpu_single_next, GRID_DIM_Q2, MAX_ITERATIONS_Q2);
    end_time = omp_get_wtime();
    std::cout << "Info: Time taken (CPU single-threaded laplace_solver): " << (end_time - start_time) * 1000.0 << " ms." << std::endl;

    // --- CPU OMP Execution ---
    initialize_grid_laplace_q2(h_grid_cpu_omp, GRID_DIM_Q2);
    initialize_grid_laplace_q2(h_grid_cpu_omp_next, GRID_DIM_Q2); // Also init boundaries for next buffer

    start_time = omp_get_wtime();
    laplace_solver_cpu_omp_q2(h_grid_cpu_omp, h_grid_cpu_omp_next, GRID_DIM_Q2, MAX_ITERATIONS_Q2);
    end_time = omp_get_wtime();
    std::cout << "Info: Time taken (CPU OMP laplace_solver): " << (end_time - start_time) * 1000.0 << " ms." << std::endl;
    
    // Correctness check for OMP against single-threaded (optional, but good practice)
    if (verify_laplace_results_q2(h_grid_cpu_single, h_grid_cpu_omp, GRID_DIM_Q2, ERROR_THRESHOLD_Q2 * 10, max_diff_val, avg_diff_val)) { // Slightly looser for OMP due to potential reordering
        std::cout << "Info: Correctness check OMP vs Single-CPU: Passed (Max Diff: " << max_diff_val << ", Avg Diff: " << avg_diff_val << ")" << std::endl;
    } else {
        std::cout << "Info: Correctness check OMP vs Single-CPU: Failed (Max Diff: " << max_diff_val << ", Avg Diff: " << avg_diff_val << ")" << std::endl;
    }

    // --- CUDA Execution ---
    float* d_current_grid, * d_next_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_current_grid, grid_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_next_grid, grid_size_bytes));

    // Initialize grid on host and copy to both device buffers (to ensure boundaries are set in both)
    initialize_grid_laplace_q2(h_grid_gpu_result, GRID_DIM_Q2); 
    CUDA_CHECK(cudaMemcpy(d_current_grid, h_grid_gpu_result, grid_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_grid, h_grid_gpu_result, grid_size_bytes, cudaMemcpyHostToDevice)); 

    dim3 threadsPerBlock(8, 8); 
    dim3 numBlocks((GRID_DIM_Q2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (GRID_DIM_Q2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));
    for (int iter = 0; iter < MAX_ITERATIONS_Q2; ++iter) {
        laplace_kernel_cuda_q2<<<numBlocks, threadsPerBlock>>>(d_current_grid, d_next_grid, GRID_DIM_Q2);
        CUDA_CHECK(cudaGetLastError()); 

        // Swap pointers for the next iteration
        float* temp = d_current_grid;
        d_current_grid = d_next_grid;
        d_next_grid = temp;
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds_gpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_gpu, start_event, stop_event));
    std::cout << "Info: Time taken (CUDA laplace_solver): " << milliseconds_gpu << " ms." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_grid_gpu_result, d_current_grid, grid_size_bytes, cudaMemcpyDeviceToHost));

    // Correctness Check (Compare CUDA result with CPU single-threaded result)
    if (verify_laplace_results_q2(h_grid_cpu_single, h_grid_gpu_result, GRID_DIM_Q2, ERROR_THRESHOLD_Q2, max_diff_val, avg_diff_val)) {
        std::cout << "Info: Correctness check CUDA laplace_solver: Passed (Max Abs Diff: " << max_diff_val 
                  << ", Avg Abs Diff: " << avg_diff_val << ")" << std::endl;
    } else {
        std::cout << "Info: Correctness check CUDA laplace_solver: Failed (Max Abs Diff: " << max_diff_val 
                  << ", Avg Abs Diff: " << avg_diff_val << ")" << std::endl;
    }
    
    std::cout << "Info: Sample values - CPU center: " << h_grid_cpu_single[(GRID_DIM_Q2/2)*GRID_DIM_Q2 + (GRID_DIM_Q2/2)]
              << ", GPU center: " << h_grid_gpu_result[(GRID_DIM_Q2/2)*GRID_DIM_Q2 + (GRID_DIM_Q2/2)] << std::endl;

    
    // Cleanup
    free(h_grid_cpu_single);
    free(h_grid_cpu_single_next);
    free(h_grid_cpu_omp);
    free(h_grid_cpu_omp_next);
    free(h_grid_gpu_result);

    CUDA_CHECK(cudaFree(d_current_grid));
    CUDA_CHECK(cudaFree(d_next_grid));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return 0;
}

