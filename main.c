#include "kernel_loader.h"

//Verzió: 0.0.1

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

int RandRange(int Min, int Max)
{
    int diff = Max-Min;
    return (int) (((double)(diff+1)/RAND_MAX) * rand() + Min);
}

int transzponalas(cl_platform_id platform_id, cl_uint n_devices, cl_device_id device_id, cl_context context, cl_program program)
{
    cl_int err;
    int error_code;

    // Create the host buffer and initialize it

    const int numRowsAndColumns = 6;

    const cl_int hostBufferSize = sizeof(cl_int) * (numRowsAndColumns * numRowsAndColumns);

    cl_int *host_buffer = malloc(hostBufferSize);

    for(int it = 0; it < numRowsAndColumns * numRowsAndColumns; it++)
    {
        host_buffer[it] = RandRange(50, 150);
    }
    
    printf("[TRANSZPONALAS-BEFORE] Az eredeti matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", host_buffer[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        hostBufferSize,
        host_buffer,
        &err
    );
    if (err != CL_SUCCESS) {
        printf("Unable to create buffer! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "transzponalas", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&numRowsAndColumns);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, 0, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer,
        CL_FALSE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 1;
    size_t n_work_groups = numRowsAndColumns * numRowsAndColumns;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer,
        CL_TRUE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    printf("[TRANSZPONALAS-AFTER] A modositott matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", host_buffer[rowIt * numRowsAndColumns + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }

    // Release the resources
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);

    free(host_buffer);

    return 1;
}

int sorosszeg_szamitas(cl_platform_id platform_id, cl_uint n_devices, cl_device_id device_id, cl_context context, cl_program program)
{
    cl_int err;
    int error_code;

    // Create the host buffer and initialize it

    const int numRowsAndColumns = 6;

    const cl_int hostBufferSize = sizeof(cl_int) * (numRowsAndColumns * numRowsAndColumns);

    cl_int *host_buffer = malloc(hostBufferSize);

    for(int it = 0; it < numRowsAndColumns * numRowsAndColumns; it++)
    {
        host_buffer[it] = RandRange(50, 150);
    }
    
    printf("[SOROSSZEG] Az eredeti matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", host_buffer[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }

    const cl_int sumBufferSize = sizeof(cl_int) * numRowsAndColumns;
    cl_int *sum_buffer = malloc(sumBufferSize);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        sum_buffer[rowIt] = 0;
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        hostBufferSize,
        host_buffer,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("Unable to create buffer! Code: %d\n", err);
        return 0;
    }

    // Create the output buffer
    cl_mem output_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sumBufferSize,
        sum_buffer,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("Unable to create buffer (#2)! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "sorosszeg_szamitas", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&numRowsAndColumns);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer,
        CL_FALSE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Sum buffer (host) -> Sum buffer (device)
    clEnqueueWriteBuffer(
        command_queue,
        output_buffer,
        CL_FALSE,
        0,
        sumBufferSize,
        sum_buffer,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 1;
    size_t global_work_size = numRowsAndColumns * numRowsAndColumns;

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer,
        CL_TRUE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Sum host buffer <- Sum device buffer
    clEnqueueReadBuffer(
        command_queue,
        output_buffer,
        CL_TRUE,
        0,
        sumBufferSize,
        sum_buffer,
        0,
        NULL,
        NULL
    );

    for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
    {   
        printf("=> A(z) %d. sor osszege: %d\n", dataIt + 1, sum_buffer[dataIt]);
    }
    
    // Release the resources
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);

    free(host_buffer);

    return 1;
}

int oszloposszeg_szamitas(cl_platform_id platform_id, cl_uint n_devices, cl_device_id device_id, cl_context context, cl_program program)
{
    cl_int err;
    int error_code;

    // Create the host buffer and initialize it

    const int numRowsAndColumns = 6;

    const cl_int hostBufferSize = sizeof(cl_int) * (numRowsAndColumns * numRowsAndColumns);

    cl_int *host_buffer = malloc(hostBufferSize);

    for(int it = 0; it < numRowsAndColumns * numRowsAndColumns; it++)
    {
        host_buffer[it] = RandRange(50, 150);
    }
    
    printf("[OSZLOPOSSZEG] Az eredeti matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", host_buffer[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }

    const cl_int sumBufferSize = sizeof(cl_int) * numRowsAndColumns;
    cl_int *sum_buffer = malloc(sumBufferSize);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        sum_buffer[rowIt] = 0;
    }

    // Create the device buffer
    cl_mem device_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        hostBufferSize,
        host_buffer,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("Unable to create buffer! Code: %d\n", err);
        return 0;
    }

    // Create the output buffer
    cl_mem output_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sumBufferSize,
        sum_buffer,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("Unable to create buffer (#2)! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "oszloposszeg_szamitas", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&numRowsAndColumns);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer,
        CL_FALSE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Sum buffer (host) -> Sum buffer (device)
    clEnqueueWriteBuffer(
        command_queue,
        output_buffer,
        CL_FALSE,
        0,
        sumBufferSize,
        sum_buffer,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 1;
    size_t global_work_size = numRowsAndColumns * numRowsAndColumns;

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer,
        CL_TRUE,
        0,
        hostBufferSize,
        host_buffer,
        0,
        NULL,
        NULL
    );

    // Sum host buffer <- Sum device buffer
    clEnqueueReadBuffer(
        command_queue,
        output_buffer,
        CL_TRUE,
        0,
        sumBufferSize,
        sum_buffer,
        0,
        NULL,
        NULL
    );

    for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
    {   
        printf("=> A(z) %d. oszlop osszege: %d\n", dataIt + 1, sum_buffer[dataIt]);
    }
    
    // Release the resources
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);

    free(host_buffer);

    return 1;
}

int matrix_szorzas(cl_platform_id platform_id, cl_uint n_devices, cl_device_id device_id, cl_context context, cl_program program, int *matrix1, int matrix1size, int *matrix2, int matrix2size)
{
    cl_int err;
    int error_code;

    // Check matrix sizes

    if(matrix1size != matrix2size)
    {
        return -2;
    }

    const size_t numRowsAndColumns = matrix1size;

    /*
    printf("[MATRIX_SZORZAS] Az eredeti 1. matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", matrix1[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }

    printf("[MATRIX_SZORZAS] Az eredeti 2. matrix %d soros:\n", numRowsAndColumns);

    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", matrix2[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }
    */

    printf("[MATRIX_SZORZAS] Before multiplicationBufferSize\n");

    const int multiplicationBufferSize = sizeof(int) * (numRowsAndColumns * numRowsAndColumns);
    int *multiplication_buffer = malloc(multiplicationBufferSize);

    printf("[MATRIX_SZORZAS] multiplicationBufferSize: %d\n", multiplicationBufferSize);

    for(int rowIt = 0; rowIt < numRowsAndColumns * numRowsAndColumns; rowIt++)
    {
        multiplication_buffer[rowIt] = 0;
    }

    printf("[MATRIX_SZORZAS] Before clCreateBuffer\n");

    // Create matrix_1_buffer
    cl_mem matrix_1_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        matrix1size * matrix1size * sizeof(int),
        matrix1,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("[MATRIX_SZORZAS] Unable to create matrix 1 buffer! Code: %d\n", err);
        return 0;
    }

    // Create matrix_2_buffer
    cl_mem matrix_2_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        matrix2size * matrix2size * sizeof(int),
        matrix2,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("[MATRIX_SZORZAS] Unable to create matrix 2 buffer! Code: %d\n", err);
        return 0;
    }

    // Create the output buffer
    cl_mem output_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        multiplicationBufferSize,
        multiplication_buffer,
        &err
    );

    if (err != CL_SUCCESS) {
        printf("[MATRIX_SZORZAS] Unable to create output buffer! Code: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "multiplication", NULL);

    // Size specification
    size_t max_work_group_size_by_device = 0;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size_by_device, NULL);

    size_t max_work_group_size_by_kernel = 0;
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size_by_kernel, NULL);

    size_t max_work_group_size_by_kernel_default = max_work_group_size_by_kernel;

    size_t global_work_size = numRowsAndColumns * (numRowsAndColumns * numRowsAndColumns);

    while(global_work_size % max_work_group_size_by_kernel != 0 && max_work_group_size_by_kernel > 1)
    {
        max_work_group_size_by_kernel--;
    }

    size_t local_work_size = fmin(fmin(max_work_group_size_by_kernel, max_work_group_size_by_device), global_work_size);

    printf("[MATRIX_SZORZAS] global_work_size: %u|local_work_size: %d|max_work_group_size_by_device: %d|max_work_group_size_by_kernel_default: %d|max_work_group_size_by_kernel_adjusted: %d|numRowsAndColumns: %d\n", global_work_size, local_work_size, max_work_group_size_by_device, max_work_group_size_by_kernel_default, max_work_group_size_by_kernel, numRowsAndColumns);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    clock_t beginKernel = clock();

    printf("[MATRIX_SZORZAS] Starting kernel...\n");

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&matrix_1_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&matrix_2_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&matrix1size);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&matrix2size);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&output_buffer);

    // Apply the kernel on the range
    cl_event event = NULL;
    cl_int ret = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );

    if(ret != CL_SUCCESS)
    {
        printf("[MATRIX_SZORZAS] clEnqueueNDRangeKernel ret: %d\n", ret);

        return 0;
    }

    clFinish(command_queue);

    clock_t endKernel = clock();

    // Multiplication device buffer -> Multiplication host buffer
    clEnqueueReadBuffer(
        command_queue,
        output_buffer,
        CL_TRUE,
        0,
        multiplicationBufferSize,
        multiplication_buffer,
        0,
        NULL,
        NULL
    );

    clock_t endBuffer = clock();


    printf("[MATRIX_SZORZAS] Kernel futasi ido: %f s|buffer copy futasi ido: %f s\n", (double)(endKernel - beginKernel) / CLOCKS_PER_SEC, (double)(endBuffer - endKernel) / CLOCKS_PER_SEC);
    printf("[MATRIX_SZORZAS] A szorzat eredmenymatrixa %d soros:\n", numRowsAndColumns);

    /*
    for(int rowIt = 0; rowIt < numRowsAndColumns; rowIt++)
    {
        printf("==> Az %d. sor szamai: ", rowIt + 1);

        for(int dataIt = 0; dataIt < numRowsAndColumns; dataIt++)
        {
            printf("%d%s", multiplication_buffer[(rowIt * numRowsAndColumns) + dataIt], (dataIt < (numRowsAndColumns - 1) ? ", " : ""));
        }

        printf("\n");
    }
    */
    
    // Release the resources
    clReleaseMemObject(matrix_1_buffer);
    clReleaseMemObject(matrix_2_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);

    return 1;
}

int main(void)
{
    srand(time(NULL));

    int i;
    cl_int err;
    int error_code;

    // Get platform
    cl_uint n_platforms;
	cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
		return 0;
	}

    // Get device
	cl_device_id device_id;
	cl_uint n_devices;
	err = clGetDeviceIDs(
		platform_id,
		CL_DEVICE_TYPE_GPU,
		1,
		&device_id,
		&n_devices
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
		return 0;
	}

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char* kernel_code = load_kernel_source("kernels/sample.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    const char options[] = "";
    err = clBuildProgram(
        program,
        1,
        &device_id,
        options,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );
        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );
        build_log[real_size] = 0;
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        return 0;
    }

    //transzponalas(platform_id, n_devices, device_id, context, program);

    //oszloposszeg_szamitas(platform_id, n_devices, device_id, context, program);

    //sorosszeg_szamitas(platform_id, n_devices, device_id, context, program);

    /* 3x3-as példa */

    /*
    cl_int matrixSizes = 3;
    cl_int *matrix1 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));
    cl_int *matrix2 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));

    matrix1[0] = 2;
    matrix1[1] = 3;
    matrix1[2] = 4;
    matrix1[3] = 5;
    matrix1[4] = 6;
    matrix1[5] = 7;
    matrix1[6] = 8;
    matrix1[7] = 9;
    matrix1[8] = 10;

    matrix2[0] = 11;
    matrix2[1] = 12;
    matrix2[2] = 13;
    matrix2[3] = 14;
    matrix2[4] = 15;
    matrix2[5] = 16;
    matrix2[6] = 17;
    matrix2[7] = 18;
    matrix2[8] = 19;
    */
    /* 3x3-as példa END */

    /* 2x2-es példa */
    /*
    cl_int matrixSizes = 2;
    cl_int *matrix1 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));
    cl_int *matrix2 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));

    matrix1[0] = 2;
    matrix1[1] = 3;
    matrix1[2] = 4;
    matrix1[3] = 5;

    matrix2[0] = 6;
    matrix2[1] = 7;
    matrix2[2] = 8;
    matrix2[3] = 9;

    matrix_szorzas(platform_id, n_devices, device_id, context, program, matrix1, matrixSizes, matrix2, matrixSizes);
    */
    /* 2x2-es példa END */

    /* 4x4-es példa */
    int matrixSizes = 8500;

    int matrixAllocationSizes = matrixSizes * matrixSizes * sizeof(int);

    int *matrix1 = malloc(matrixAllocationSizes);
    int *matrix2 = malloc(matrixAllocationSizes);

    for(int i = 0; i < matrixSizes * matrixSizes; i++)
    {
        matrix1[i] = RandRange(10, 20);
        matrix2[i] = RandRange(10, 20);
    }

    /*
    cl_int matrixSizes = 4;
    cl_int *matrix1 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));
    cl_int *matrix2 = malloc(matrixSizes * matrixSizes * sizeof(cl_int));

    matrix1[0] = 2;
    matrix1[1] = 3;
    matrix1[2] = 4;
    matrix1[3] = 5;
    matrix1[4] = 6;
    matrix1[5] = 7;
    matrix1[6] = 8;
    matrix1[7] = 9;
    matrix1[8] = 10;
    matrix1[9] = 11;
    matrix1[10] = 12;
    matrix1[11] = 13;
    matrix1[12] = 14;
    matrix1[13] = 15;
    matrix1[14] = 16;
    matrix1[15] = 17;

    matrix2[0] = 18;
    matrix2[1] = 19;
    matrix2[2] = 20;
    matrix2[3] = 21;
    matrix2[4] = 22;
    matrix2[5] = 23;
    matrix2[6] = 24;
    matrix2[7] = 25;
    matrix2[8] = 26;
    matrix2[9] = 27;
    matrix2[10] = 28;
    matrix2[11] = 29;
    matrix2[12] = 30;
    matrix2[13] = 31;
    matrix2[14] = 32;
    matrix2[15] = 33;
    */
    /* 4x4-es példa END */

    matrix_szorzas(platform_id, n_devices, device_id, context, program, matrix1, matrixSizes, matrix2, matrixSizes);

    //Release resources
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(matrix1);
    free(matrix2);
}