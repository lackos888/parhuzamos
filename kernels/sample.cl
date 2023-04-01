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