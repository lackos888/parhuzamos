__kernel void transzponalas(__global int* matrix, int numRowsAndColumns)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    int rowNum = (idx / numRowsAndColumns);

    int columnNum = idx % numRowsAndColumns;

    atomic_xchg(&matrix[rowNum * numRowsAndColumns + columnNum], matrix[rowNum + columnNum * numRowsAndColumns]);

    //printf("rowNum: %d | columnNum: %d | idx: %d | cpy: %d\n", rowNum, columnNum, idx, cpy);
}

__kernel void sorosszeg_szamitas(__global int* matrix, int numRowsAndColumns, __global int *sum)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    int rowNum = (idx / numRowsAndColumns);

    int columnNum = idx % numRowsAndColumns;

    atomic_add(&sum[rowNum], matrix[rowNum * numRowsAndColumns + columnNum]);
}

__kernel void oszloposszeg_szamitas(__global int* matrix, int numRowsAndColumns, __global int *sum)
{
    size_t idx = get_global_id(0);

    if(idx > numRowsAndColumns * numRowsAndColumns)
    {
        return;
    }

    int rowNum = (idx / numRowsAndColumns);

    int columnNum = idx % numRowsAndColumns;

    atomic_add(&sum[rowNum], matrix[rowNum + columnNum * numRowsAndColumns]);
}

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
    size_t maxIterations = matrix1Size * matrix1Size * matrix1Size;

    size_t idxReal = get_global_id(0);
    size_t startIdx = idxReal * MATRIX_MUL_BLOCK_SIZE;

    if(startIdx > maxIterations)
    {
        return;
    }

    for(int workIt = 0; workIt < MATRIX_MUL_BLOCK_SIZE; workIt++)
    {
        size_t idx = startIdx + workIt;

        if(idx > maxIterations)
        {
            break;
        }

        size_t realIdx = idx;

        calculatedOff str = {0};
        idx = calculate_offset_and_other_things(idx, matrix1Size, &str);

        size_t offset = str.offset;
        size_t numberOfSubtractionsDone = str.numberOfSubtractionsDone;

        int inputRowNumFromMatrix1 = min(matrix1Size, (int)((idx / matrix1Size) + offset));

        int inputColumnNumFromMatrix1 = (idx % matrix1Size);

        int inputRowNumFromMatrix2 = inputColumnNumFromMatrix1;

        int inputColumnNumFromMatrix2 = inputRowNumFromMatrix1 + (numberOfSubtractionsDone - offset);

        long multiplyA = matrix1[(inputRowNumFromMatrix1 * matrix1Size) + inputColumnNumFromMatrix1];

        long multiplyB = matrix2[(inputRowNumFromMatrix2 * matrix2Size) + inputColumnNumFromMatrix2];

        int outputRowNum = min(matrix1Size, (int)((idx / matrix1Size) + offset));

        int outputRowColumn = numberOfSubtractionsDone;

        //printf("realIdx: %d|idx: %d|(%d, %d)|%d\n", realIdx, idx, outputRowNum, outputRowColumn, sumRetOutputIdx);

        atomic_add(&mul[outputRowNum * matrix1Size + outputRowColumn], multiplyA * multiplyB);
    }

    //printf("idx %d: %d*%d|output: (%d, %d)|input: (%d, %d)*(%d, %d)|offset: %d|fakeIdx: %d|numberOfSubtractionsDone: %d\n", realIdx, multiplyA, multiplyB, outputRowNum, outputRowColumn, inputRowNumFromMatrix1, inputColumnNumFromMatrix1, inputRowNumFromMatrix2, inputColumnNumFromMatrix2, offset, idx, numberOfSubtractionsDone); 
}