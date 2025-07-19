%%writefile q2_laplace_solver_opencl.cpp
 // If using Colab with OpenCL setup
#define CL_TARGET_OPENCL_VERSION 200 // Or 120, 300 as per your SDK
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// --- OpenCL Error Checking ---
const char* clGetErrorStringDetailQ2(cl_int error) { /* ... (same as in Q1 OpenCL) ... */
    static const char* errorString[] = {
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE", "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE", "CL_MEM_COPY_OVERLAP", "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED", "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE", 
        "CL_MISALIGNED_SUB_BUFFER_OFFSET", "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", "CL_COMPILE_PROGRAM_FAILURE",
        "CL_LINKER_NOT_AVAILABLE", "CL_LINK_PROGRAM_FAILURE", "CL_DEVICE_PARTITION_FAILED",
        "CL_KERNEL_ARG_INFO_NOT_AVAILABLE", "", "", "", "", "", "",
        "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE", "CL_INVALID_PLATFORM", "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT", "CL_INVALID_QUEUE_PROPERTIES", "CL_INVALID_COMMAND_QUEUE", "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT", "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", "CL_INVALID_IMAGE_SIZE", "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY", "CL_INVALID_BUILD_OPTIONS", "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME", "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL", "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE", "CL_INVALID_ARG_SIZE", "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE", "CL_INVALID_WORK_ITEM_SIZE", "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST", "CL_INVALID_EVENT", "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE", "CL_INVALID_MIP_LEVEL", "CL_INVALID_GLOBAL_WORK_SIZE"
    };
    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
    const int index = -error;
    return (index >= 0 && index < errorCount && errorString[index][0] != '\0') ? errorString[index] : "Unknown OpenCL error";
}
#define CL_CHECK_Q2(err_code) { \
    if (err_code != CL_SUCCESS) { \
        std::cerr << "OpenCL Error (" << __FILE__ << ":" << __LINE__ << "): Code " \
                  << err_code << " - " << clGetErrorStringDetailQ2(err_code) << std::endl; \
        /* exit(EXIT_FAILURE); */ \
    } \
}

// --- Parameters for Laplace Q2 ---
const int GRID_DIM_Q2_OCL = 512;
const int MAX_ITERATIONS_Q2_OCL = 1000;

const float TOP_VOLTAGE_Q2_OCL = 5.0f;
const float BOTTOM_VOLTAGE_Q2_OCL = -5.0f;
const float LEFT_VOLTAGE_Q2_OCL = 0.0f;
const float RIGHT_VOLTAGE_Q2_OCL = 0.0f;
const float INITIAL_GUESS_Q2_OCL = 0.0f;
const float ERROR_THRESHOLD_Q2_OCL = 1e-3f;

// --- OpenCL Kernel String for Laplace Solver ---
const char* laplace_kernel_source_q2 = R"(
__kernel void laplace_iter_ocl(
    __global const float* current_grid,
    __global float* next_grid,
    const int N // Grid dimension (width and height)
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    // Only update interior points
    if (row > 0 && row < N - 1 && col > 0 && col < N - 1) {
        next_grid[row * N + col] = 0.25f * (
            current_grid[(row - 1) * N + col] +  // Top
            current_grid[(row + 1) * N + col] +  // Bottom
            current_grid[row * N + (col - 1)] +  // Left
            current_grid[row * N + (col + 1)]    // Right
        );
    } else if (row < N && col < N) { 
        // For boundary points, or points outside strict interior being processed by a block,
        // copy current to next to preserve boundary values.
        // This is important because the kernel is launched over the entire grid.
        next_grid[row * N + col] = current_grid[row * N + col];
    }
}
)";

// --- Initialize Grid (same as CUDA Q2) ---
void initialize_grid_laplace_q2(float* grid, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0) grid[i * N + j] = TOP_VOLTAGE_Q2_OCL;
            else if (i == N - 1) grid[i * N + j] = BOTTOM_VOLTAGE_Q2_OCL;
            else if (j == 0) grid[i * N + j] = LEFT_VOLTAGE_Q2_OCL;
            else if (j == N - 1) grid[i * N + j] = RIGHT_VOLTAGE_Q2_OCL;
            else grid[i * N + j] = INITIAL_GUESS_Q2_OCL;
        }
    }
}

// --- CPU Single-Threaded Laplace (same as CUDA Q2) ---
void laplace_solver_cpu_single_q2(float* current_grid, float* next_grid, int N, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next_grid[i * N + j] = 0.25f * (
                    current_grid[(i - 1) * N + j] + current_grid[(i + 1) * N + j] +
                    current_grid[i * N + (j - 1)] + current_grid[i * N + (j + 1)]
                );
            }
        }
        for (int i = 1; i < N - 1; ++i) { // Copy back interior
            for (int j = 1; j < N - 1; ++j) {
                current_grid[i * N + j] = next_grid[i * N + j];
            }
        }
    }
}

// --- CPU OMP Laplace (same as CUDA Q2) ---
void laplace_solver_cpu_omp_q2(float* current_grid, float* next_grid, int N, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next_grid[i * N + j] = 0.25f * (
                    current_grid[(i - 1) * N + j] + current_grid[(i + 1) * N + j] +
                    current_grid[i * N + (j - 1)] + current_grid[i * N + (j + 1)]
                );
            }
        }
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                current_grid[i * N + j] = next_grid[i * N + j];
            }
        }
    }
}

// --- Correctness Check (same as CUDA Q2) ---
bool verify_laplace_results_q2(const float* ref_grid, const float* other_grid, int N, float threshold, float& max_abs_diff, float& avg_abs_diff) {
    max_abs_diff = 0.0f;
    double sum_abs_diff = 0.0;
    int interior_points_count = 0;
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float diff = fabs(ref_grid[i * N + j] - other_grid[i * N + j]);
            if (diff > max_abs_diff) max_abs_diff = diff;
            sum_abs_diff += diff;
            interior_points_count++;
        }
    }
    avg_abs_diff = (interior_points_count > 0) ? static_cast<float>(sum_abs_diff / interior_points_count) : 0.0f;
    return max_abs_diff <= threshold;
}

// Helper to get OpenCL device (same as Q1 OpenCL)
cl_device_id get_opencl_device_q2() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int err;

    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (err != CL_SUCCESS || ret_num_platforms == 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl; return NULL;
    }
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS || ret_num_devices == 0) {
        std::cout << "No GPU device found, trying CPU for OpenCL." << std::endl;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        if (err != CL_SUCCESS || ret_num_devices == 0) {
            std::cerr << "Failed to find any OpenCL GPU or CPU devices." << std::endl; return NULL;
        }
    }
    char deviceName[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    std::cout << "Info: Using OpenCL device: " << deviceName << std::endl;
    return device_id;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    size_t grid_elements_q2 = (size_t)GRID_DIM_Q2_OCL * GRID_DIM_Q2_OCL;
    size_t grid_size_bytes_q2 = grid_elements_q2 * sizeof(float);

    float* h_grid_cpu_single = (float*)malloc(grid_size_bytes_q2);
    float* h_grid_cpu_single_next = (float*)malloc(grid_size_bytes_q2);
    float* h_grid_cpu_omp = (float*)malloc(grid_size_bytes_q2);
    float* h_grid_cpu_omp_next = (float*)malloc(grid_size_bytes_q2);
    float* h_grid_ocl_result = (float*)malloc(grid_size_bytes_q2);

    if (!h_grid_cpu_single || !h_grid_cpu_single_next || !h_grid_cpu_omp || !h_grid_cpu_omp_next || !h_grid_ocl_result) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        // Basic cleanup
        free(h_grid_cpu_single); free(h_grid_cpu_single_next); free(h_grid_cpu_omp); free(h_grid_cpu_omp_next); free(h_grid_ocl_result);
        return 1;
    }

    double start_time, end_time;
    float max_diff_val, avg_diff_val;

    // --- CPU Single-Threaded ---
    initialize_grid_laplace_q2(h_grid_cpu_single, GRID_DIM_Q2_OCL);
    initialize_grid_laplace_q2(h_grid_cpu_single_next, GRID_DIM_Q2_OCL); // Init temp buffer too
    start_time = omp_get_wtime();
    laplace_solver_cpu_single_q2(h_grid_cpu_single, h_grid_cpu_single_next, GRID_DIM_Q2_OCL, MAX_ITERATIONS_Q2_OCL);
    end_time = omp_get_wtime();
    std::cout << "Info: Time taken (CPU single-threaded laplace_solver): " << (end_time - start_time) * 1000.0 << " ms." << std::endl;

    // --- CPU OMP ---
    initialize_grid_laplace_q2(h_grid_cpu_omp, GRID_DIM_Q2_OCL);
    initialize_grid_laplace_q2(h_grid_cpu_omp_next, GRID_DIM_Q2_OCL);
    start_time = omp_get_wtime();
    laplace_solver_cpu_omp_q2(h_grid_cpu_omp, h_grid_cpu_omp_next, GRID_DIM_Q2_OCL, MAX_ITERATIONS_Q2_OCL);
    end_time = omp_get_wtime();
    std::cout << "Info: Time taken (CPU OMP laplace_solver): " << (end_time - start_time) * 1000.0 << " ms." << std::endl;
    // (Optional: Verify OMP vs Single CPU)

    // --- OpenCL Execution ---
    cl_int err;
    cl_device_id device_id = get_opencl_device_q2();
    if (!device_id) return 1;

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err); CL_CHECK_Q2(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err); CL_CHECK_Q2(err);

    // Create two buffers for ping-ponging on the device
    cl_mem d_grid_A = clCreateBuffer(context, CL_MEM_READ_WRITE, grid_size_bytes_q2, NULL, &err); CL_CHECK_Q2(err);
    cl_mem d_grid_B = clCreateBuffer(context, CL_MEM_READ_WRITE, grid_size_bytes_q2, NULL, &err); CL_CHECK_Q2(err);

    // Initialize host grid for OpenCL and copy to d_grid_A
    initialize_grid_laplace_q2(h_grid_ocl_result, GRID_DIM_Q2_OCL); // Use h_grid_ocl_result for initial host data
    CL_CHECK_Q2(clEnqueueWriteBuffer(queue, d_grid_A, CL_TRUE, 0, grid_size_bytes_q2, h_grid_ocl_result, 0, NULL, NULL));
    // Also copy initial state to d_grid_B as its boundaries will be preserved by kernel
    CL_CHECK_Q2(clEnqueueWriteBuffer(queue, d_grid_B, CL_TRUE, 0, grid_size_bytes_q2, h_grid_ocl_result, 0, NULL, NULL));

    cl_program program = clCreateProgramWithSource(context, 1, &laplace_kernel_source_q2, NULL, &err); CL_CHECK_Q2(err);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL Program Build Log:\n" << log.data() << std::endl;
        CL_CHECK_Q2(err);
    }
    cl_kernel kernel = clCreateKernel(program, "laplace_iter_ocl", &err); CL_CHECK_Q2(err);
    CL_CHECK_Q2(clSetKernelArg(kernel, 2, sizeof(cl_int), &GRID_DIM_Q2_OCL)); // Arg N

    size_t global_work_size[2] = {(size_t)GRID_DIM_Q2_OCL, (size_t)GRID_DIM_Q2_OCL};
    size_t local_work_size[2] = {8, 8}; // 8x8 workgroup size

    cl_mem d_current = d_grid_A;
    cl_mem d_next = d_grid_B;

    cl_event ocl_iter_event; // Event to time iterations
    auto ocl_total_start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITERATIONS_Q2_OCL; ++iter) {
        CL_CHECK_Q2(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_current));
        CL_CHECK_Q2(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_next));
        
        CL_CHECK_Q2(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, (iter == MAX_ITERATIONS_Q2_OCL - 1) ? &ocl_iter_event : NULL));
        
        // Swap buffers for next iteration
        cl_mem temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
    CL_CHECK_Q2(clWaitForEvents(1, &ocl_iter_event)); // Wait for the last kernel to finish
    auto ocl_total_end_time = std::chrono::high_resolution_clock::now();
    double ocl_total_execution_time_ms = std::chrono::duration<double, std::milli>(ocl_total_end_time - ocl_total_start_time).count();
    std::cout << "Info: Time taken (OpenCL laplace_solver Host Timer): " << ocl_total_execution_time_ms << " ms." << std::endl;

    CL_CHECK_Q2(clEnqueueReadBuffer(queue, d_current, CL_TRUE, 0, grid_size_bytes_q2, h_grid_ocl_result, 0, NULL, NULL)); // Result is in d_current after last swap

    // Correctness Check
    if (verify_laplace_results_q2(h_grid_cpu_single, h_grid_ocl_result, GRID_DIM_Q2_OCL, ERROR_THRESHOLD_Q2_OCL, max_diff_val, avg_diff_val)) {
        std::cout << "Info: Correctness check laplace_solver (OpenCL vs CPU Single): Passed (Max Abs Diff: " << max_diff_val << ", Avg Abs Diff: " << avg_diff_val << ")" << std::endl;
    } else {
        std::cout << "Info: Correctness check laplace_solver (OpenCL vs CPU Single): Failed (Max Abs Diff: " << max_diff_val << ", Avg Abs Diff: " << avg_diff_val << ")" << std::endl;
    }
    std::cout << "Info: Sample values - CPU center: " << h_grid_cpu_single[(GRID_DIM_Q2_OCL/2)*GRID_DIM_Q2_OCL + (GRID_DIM_Q2_OCL/2)]
              << ", OpenCL center: " << h_grid_ocl_result[(GRID_DIM_Q2_OCL/2)*GRID_DIM_Q2_OCL + (GRID_DIM_Q2_OCL/2)] << std::endl;

    // Cleanup
    CL_CHECK_Q2(clReleaseEvent(ocl_iter_event));
    CL_CHECK_Q2(clReleaseMemObject(d_grid_A));
    CL_CHECK_Q2(clReleaseMemObject(d_grid_B));
    CL_CHECK_Q2(clReleaseKernel(kernel));
    CL_CHECK_Q2(clReleaseProgram(program));
    CL_CHECK_Q2(clReleaseCommandQueue(queue));
    CL_CHECK_Q2(clReleaseContext(context));

    free(h_grid_cpu_single); free(h_grid_cpu_single_next);
    free(h_grid_cpu_omp); free(h_grid_cpu_omp_next);
    free(h_grid_ocl_result);

    return 0;
}
