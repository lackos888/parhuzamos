__kernel void transzponalas(__global long* matrix, int numRowsAndColumns)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    size_t rowNum = (idx / numRowsAndColumns);

    size_t columnNum = idx % numRowsAndColumns;

    atomic_xchg(&matrix[rowNum * numRowsAndColumns + columnNum], matrix[rowNum + columnNum * numRowsAndColumns]);
}

__kernel void sorosszeg_szamitas(__global int* matrix, int numRowsAndColumns, __global int *sum)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    size_t rowNum = (idx / numRowsAndColumns);

    size_t columnNum = idx % numRowsAndColumns;

    atomic_add(&sum[rowNum], matrix[rowNum * numRowsAndColumns + columnNum]);
}

__kernel void oszloposszeg_szamitas(__global int* matrix, int numRowsAndColumns, __global int *sum)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    size_t rowNum = (idx / numRowsAndColumns);

    size_t columnNum = idx % numRowsAndColumns;

    atomic_add(&sum[rowNum], matrix[rowNum + columnNum * numRowsAndColumns]);
}

#define LOCAL_BUFFER_SIZE 1024

typedef struct 
{
    size_t offset;
    size_t numberOfSubtractionsDone;
} calculatedOff;

size_t calculate_offset_and_other_things(size_t idx, int matrix1Size, calculatedOff *res)
{
    size_t secondPow = (matrix1Size * matrix1Size);

    size_t offset = (idx / secondPow);

    idx = (idx % (secondPow));

    size_t numberOfSubtractionsDone = (idx / matrix1Size);

    idx = (idx % matrix1Size); 

    res->offset = offset;
    res->numberOfSubtractionsDone = numberOfSubtractionsDone;

    return idx;
}

__kernel void multiplication(__global long* matrix1, __global long* matrix2, int matrix1Size, int matrix2Size, __global long *mul)
{
    __local long sumRet[LOCAL_BUFFER_SIZE];

    size_t idx = get_global_id(0);
    size_t localIdx = get_local_id(0);

    if(localIdx == 0)
    {
        for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
        {
            sumRet[i] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    size_t groupSize = get_local_size(0);
    size_t groupGlobalIdxStart = idx - localIdx;
    size_t groupGlobalIdxEnd = groupGlobalIdxStart + groupSize;

    size_t realIdx = idx;

    calculatedOff offsets = {0};
    idx = calculate_offset_and_other_things(idx, matrix1Size, &offsets);

    size_t offset = offsets.offset;
    size_t numberOfSubtractionsDone = offsets.numberOfSubtractionsDone;

    size_t inputRowNumFromMatrix1 = min((size_t)matrix1Size, (size_t)(idx / matrix1Size) + offset);

    size_t inputColumnNumFromMatrix1 = (idx % matrix1Size);

    size_t inputRowNumFromMatrix2 = inputColumnNumFromMatrix1;

    size_t inputColumnNumFromMatrix2 = inputRowNumFromMatrix1 + (numberOfSubtractionsDone - offset);

    long multiplyA = matrix1[(inputRowNumFromMatrix1 * matrix1Size) + inputColumnNumFromMatrix1];

    long multiplyB = matrix2[(inputRowNumFromMatrix2 * matrix2Size) + inputColumnNumFromMatrix2];

    for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
    {
        size_t idx2 = groupGlobalIdxStart + i;

        calculatedOff offsets = {0};
        idx2 = calculate_offset_and_other_things(idx2, matrix1Size, &offsets);

        size_t offset2 = offsets.offset;
        size_t numberOfSubtractionsDone2 = offsets.numberOfSubtractionsDone;

        if(offset2 != offset || numberOfSubtractionsDone2 != numberOfSubtractionsDone)
        {
            continue;
        }

        atomic_add(&sumRet[i], multiplyA * multiplyB);

        break;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(localIdx == groupSize - 1)
    {
        for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
        {
            if(sumRet[i] == 0)
            {
                continue;
            }

            size_t idx2 = groupGlobalIdxStart + i;

            calculatedOff offsets = {0};
            idx2 = calculate_offset_and_other_things(idx2, matrix1Size, &offsets);

            size_t offset2 = offsets.offset;
            size_t numberOfSubtractionsDone2 = offsets.numberOfSubtractionsDone;

            long outputRowNum = min(matrix1Size, (int)((idx2 / matrix1Size) + offset2));

            long outputRowColumn = numberOfSubtractionsDone2;

            atomic_add(&mul[outputRowNum * matrix1Size + outputRowColumn], sumRet[i]);
        }
    }
    
    //printf("idx %d: %d*%d|output: (%d, %d)|input: (%d, %d)*(%d, %d)|offset: %d|fakeIdx: %d|numberOfSubtractionsDone: %d\n", realIdx, multiplyA, multiplyB, outputRowNum, outputRowColumn, inputRowNumFromMatrix1, inputColumnNumFromMatrix1, inputRowNumFromMatrix2, inputColumnNumFromMatrix2, offset, idx, numberOfSubtractionsDone); 
}