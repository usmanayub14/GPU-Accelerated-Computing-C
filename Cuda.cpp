
%%writefile q1_matrix_multiplication_cuda.cu
#include <iostream>
#include <vector>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()
#include <cmath>   // For fabs()
#include <iomanip> // For std::fixed, std::setprecision
#include <omp.h>   // For OpenMP and omp_get_wtime()
#include <cuda_runtime.h>

// Helper function for CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// --- Matrix Initialization ---
void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand() % 100) / 10.0f; // Random float between 0.0 and 9.9
    }
}

// --- CPU Single-Threaded Matrix Multiplication (Naive) ---
// C (M x N_dim) = A (M x K) * B (K x N_dim)
void matrix_mult_cpu_single(const float* A, const float* B, float* C, int M, int K_dim, int N_dim) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += A[i * K_dim + k] * B[k * N_dim + j];
            }
            C[i * N_dim + j] = sum;
        }
    }
}

// --- CPU OMP Matrix Multiplication (Naive) ---
// C (M x N_dim) = A (M x K) * B (K x N_dim)
void matrix_mult_cpu_omp(const float* A, const float* B, float* C, int M, int K_dim, int N_dim) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += A[i * K_dim + k] * B[k * N_dim + j];
            }
            C[i * N_dim + j] = sum;
        }
    }
}

// --- CUDA Kernel for Naive Matrix Multiplication ---
// Assumes square matrices for simplicity in this kernel C = A * B, all N_dim x N_dim
__global__ void matrix_mult_kernel_cuda(const float* A, const float* B, float* C, int N_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N_dim && col < N_dim) {
        float sum = 0.0f;
        for (int k = 0; k < N_dim; ++k) {
            sum += A[row * N_dim + k] * B[k * N_dim + col];
        }
        C[row * N_dim + col] = sum;
    }
}

// --- Correctness Check ---
bool verify_results(const float* ref_C, const float* other_C, int rows, int cols, float threshold, float& max_diff) {
    max_diff = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabs(ref_C[i] - other_C[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > threshold) {
            // Uncomment for detailed failure info:
            // std::cerr << "Verification failed at index " << i 
            //           << ": Ref=" << ref_C[i] << ", Other=" << other_C[i] 
            //           << ", Diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    srand(static_cast<unsigned int>(time(0)));

    // For naive matrix multiplication, let's use square matrices.
    // N x N matrix multiplication: A(N x N) * B(N x N) = C(N x N)
    const int N_MATRIX_DIM = 1024; // Dimension for square matrices

    const float ERROR_THRESHOLD_Q1 = 1e-2f; // Threshold for floating point comparison

    // Host memory allocation
    size_t matrix_elements = (size_t)N_MATRIX_DIM * N_MATRIX_DIM;
    size_t matrix_size_bytes = matrix_elements * sizeof(float);

    float* h_A = (float*)malloc(matrix_size_bytes);
    float* h_B = (float*)malloc(matrix_size_bytes);
    float* h_C_cpu_single = (float*)malloc(matrix_size_bytes);
    float* h_C_cpu_omp = (float*)malloc(matrix_size_bytes);
    float* h_C_gpu = (float*)malloc(matrix_size_bytes);

    if (!h_A || !h_B || !h_C_cpu_single || !h_C_cpu_omp || !h_C_gpu) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        // Free any partially allocated memory
        free(h_A); free(h_B); free(h_C_cpu_single); free(h_C_cpu_omp); free(h_C_gpu);
        return 1;
    }

    // Initialize matrices A and B
    initialize_matrix(h_A, N_MATRIX_DIM, N_MATRIX_DIM);
    initialize_matrix(h_B, N_MATRIX_DIM, N_MATRIX_DIM);

    double start_time_cpu, end_time_cpu;
    float max_diff;

    std::cout << std::fixed << std::setprecision(3); // For consistent float output

    // --- CPU Single-Threaded Execution ---
    start_time_cpu = omp_get_wtime();
    matrix_mult_cpu_single(h_A, h_B, h_C_cpu_single, N_MATRIX_DIM, N_MATRIX_DIM, N_MATRIX_DIM);
    end_time_cpu = omp_get_wtime();
    std::cout << "Info: Time taken (CPU single-threaded matrix_multiplication): " << (end_time_cpu - start_time_cpu) * 1000.0 << " ms." << std::endl;

    // --- CPU OMP Execution ---
    start_time_cpu = omp_get_wtime();
    matrix_mult_cpu_omp(h_A, h_B, h_C_cpu_omp, N_MATRIX_DIM, N_MATRIX_DIM, N_MATRIX_DIM);
    end_time_cpu = omp_get_wtime();
    std::cout << "Info: Time taken (CPU OMP matrix_multiplication): " << (end_time_cpu - start_time_cpu) * 1000.0 << " ms." << std::endl;

    // --- CUDA Execution ---
    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, matrix_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, matrix_size_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8); // As specified: 8x8 workgroup size
    dim3 numBlocks((N_MATRIX_DIM + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N_MATRIX_DIM + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));
    matrix_mult_kernel_cuda<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N_MATRIX_DIM);
    CUDA_CHECK(cudaGetLastError()); // Important: Check for errors after kernel launch
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event)); // Wait for the kernel and all preceding CUDA calls to complete

    float milliseconds_gpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_gpu, start_event, stop_event));
    std::cout << "Info: Time taken (CUDA matrix_multiplication): " << milliseconds_gpu << " ms." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, matrix_size_bytes, cudaMemcpyDeviceToHost));

    // Correctness Check (Compare CUDA result with CPU single-threaded result)
    if (verify_results(h_C_cpu_single, h_C_gpu, N_MATRIX_DIM, N_MATRIX_DIM, ERROR_THRESHOLD_Q1, max_diff)) {
        std::cout << "Info: Correctness check matrix_multiplication: Passed (Max Diff: " << max_diff << ")" << std::endl;
    } else {
        std::cout << "Info: Correctness check matrix_multiplication: Failed (Max Diff: " << max_diff << ")" << std::endl;
    }


    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu_single);
    free(h_C_cpu_omp);
    free(h_C_gpu);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return 0;
}

