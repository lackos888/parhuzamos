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

__kernel void multiplication(__global int* matrices, int matrix1Size, int matrix2Size, __global int *mul, int chunkSize, int overallWorkersNum)
{
    size_t idx = get_global_id(0);
    size_t realIdx = idx;

    size_t offset = 0;

    while(idx >= matrix1Size * matrix1Size)
    {
        idx -= matrix1Size * matrix1Size;
        offset++;
    }

    size_t idxStuff = idx;
    size_t numberOfSubtractionsDone = 0;

    while(idxStuff > matrix1Size - 1)
    {
        idxStuff -= matrix1Size;

        numberOfSubtractionsDone++;
    }

    int inputRowNumFromMatrix1 = min(matrix1Size, (int)((idxStuff / matrix1Size) + offset));

    int inputColumnNumFromMatrix1 = (idxStuff % matrix1Size);

    int inputRowNumFromMatrix2 = inputColumnNumFromMatrix1;

    int inputColumnNumFromMatrix2 = inputRowNumFromMatrix1 + (numberOfSubtractionsDone - offset);

    int multiplyA = matrices[(inputRowNumFromMatrix1 * matrix1Size) + inputColumnNumFromMatrix1];

    int multiplyB = matrices[(matrix1Size * matrix1Size) + inputRowNumFromMatrix2 * matrix2Size + inputColumnNumFromMatrix2];

    int outputRowNum = min(matrix1Size, (int)((idxStuff / matrix1Size) + offset));

    int outputRowColumn = numberOfSubtractionsDone;

    atomic_add(&mul[outputRowNum * matrix1Size + outputRowColumn], multiplyA * multiplyB);
    //printf("idx %d: %d*%d|output: (%d, %d)|input: (%d, %d)*(%d, %d)|offset: %d|fakeIdx: %d|idxStuff: %d|numberOfSubtractionsDone: %d\n", realIdx, multiplyA, multiplyB, outputRowNum, outputRowColumn, inputRowNumFromMatrix1, inputColumnNumFromMatrix1, inputRowNumFromMatrix2, inputColumnNumFromMatrix2, offset, idx, idxStuff, numberOfSubtractionsDone); 
}