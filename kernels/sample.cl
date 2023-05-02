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

#define LOCAL_BUFFER_SIZE 512

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

    size_t offset = (idx / (matrix1Size * matrix1Size));

    idx = (idx % (matrix1Size * matrix1Size));

    size_t numberOfSubtractionsDone = (idx / matrix1Size);

    idx = (idx % matrix1Size);

    size_t inputRowNumFromMatrix1 = min((size_t)matrix1Size, (size_t)(idx / matrix1Size) + offset);

    size_t inputColumnNumFromMatrix1 = (idx % matrix1Size);

    size_t inputRowNumFromMatrix2 = inputColumnNumFromMatrix1;

    size_t inputColumnNumFromMatrix2 = inputRowNumFromMatrix1 + (numberOfSubtractionsDone - offset);

    long multiplyA = matrix1[(inputRowNumFromMatrix1 * matrix1Size) + inputColumnNumFromMatrix1];

    long multiplyB = matrix2[(inputRowNumFromMatrix2 * matrix2Size) + inputColumnNumFromMatrix2];

    for(int i = 0; i < LOCAL_BUFFER_SIZE; i++)
    {
        size_t idx2 = groupGlobalIdxStart + i;

        size_t offset2 = (idx2 / (matrix1Size * matrix1Size));

        idx2 = (idx2 % (matrix1Size * matrix1Size));

        size_t numberOfSubtractionsDone2 = (idx2 / matrix1Size);

        idx2 = (idx2 % matrix1Size);

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

            size_t offset2 = (idx2 / (matrix1Size * matrix1Size));

            idx2 = (idx2 % (matrix1Size * matrix1Size));

            size_t numberOfSubtractionsDone2 = (idx2 / matrix1Size);

            idx2 = (idx2 % matrix1Size);

            long outputRowNum = min(matrix1Size, (int)((idx2 / matrix1Size) + offset2));

            long outputRowColumn = numberOfSubtractionsDone2;

            atomic_add(&mul[outputRowNum * matrix1Size + outputRowColumn], sumRet[i]);
        }
    }
    
    //printf("idx %d: %d*%d|output: (%d, %d)|input: (%d, %d)*(%d, %d)|offset: %d|fakeIdx: %d|numberOfSubtractionsDone: %d\n", realIdx, multiplyA, multiplyB, outputRowNum, outputRowColumn, inputRowNumFromMatrix1, inputColumnNumFromMatrix1, inputRowNumFromMatrix2, inputColumnNumFromMatrix2, offset, idx, numberOfSubtractionsDone); 
}