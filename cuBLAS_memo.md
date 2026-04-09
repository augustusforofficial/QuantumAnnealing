## 1.1 Data Layout
- column-major(列優先) storage and 1-based indexing(1からのインデックス)
```C
#define IDX2C(i,j,ld) (((j-1)*ld) + (i-1))
```
- column-major storage and 0-based indexing
```C
#define IDX2C(i,j,ld) (((j)*ld) + (i))
```
これは, 以下のようなインデックスのつけ方になり、確かに"column-major"である。
```math
\begin{pmatrix}
0 & 4 & 8 & 12\\
1 & 5 & 9 & 13\\
2 & 6 & 10 & 14\\
3 & 7 & 11 & 15
\end{pmatrix}
```

## 2　cuBLAS API
https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api
https://intro-to-cuda.readthedocs.io/en/latest/tutorial/cuBLAS.html
- Errorは cublasStatus_t にて返される
```C
cublasreate()

cublasDestroy()
```

| type | t | Meaning| 
| --- | --- | --- | 
| float | s or S | real single-precision|
| double | d or D | real double-precision|
| cuComplex | c or C| complex single-precision|
| cuDoubleComplex | z or Z| complex double-precision|


## その他
- cuDoubleComplex は 16 B (128 bit)
