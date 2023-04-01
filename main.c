#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

    sorosszeg_szamitas(platform_id, n_devices, device_id, context, program);

    //Release resources
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
}
