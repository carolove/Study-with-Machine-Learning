# 学习buddy项目-读懂convert-memcpy-to-gpu
## 原理
```
其实就是将gpu.launch_func 涉及的矩阵入参，将memref替换为gpu.alloc&gpu.memcpy&gpu.dealloc
```
## 核心代码
```

```
## mlir演示
```
代码变更
    // add
    %memref = gpu.alloc  () : memref<5376x2048xf32>
    gpu.memcpy  %memref, %arg0 : memref<5376x2048xf32>, memref<5376x2048xf32>
    %memref_0 = gpu.alloc  () : memref<2048x5376xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<2048x5376xf32>, memref<2048x5376xf32>

    %memref_5 = gpu.alloc  () : memref<5376x5376xf32>
    gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c21, %c42, %c1) threads in (%c64, %c2, %c1)  args(%c256 : index, %c0 : index, %c-1 : index, %c32 : index, %c128 : index, %c64 : index, %cst : vector<8xf32>, %memref_5 : memref<5376x5376xf32>, %c1 : index, %c2 : index, %c3 : index, %c4 : index, %c5 : index, %c6 : index, %c7 : index, %c8 : index, %c9 : index, %c10 : index, %c11 : index, %c12 : index, %c13 : index, %c14 : index, %c15 : index, %c16 : index, %c24 : index, %c40 : index, %c48 : index, %c56 : index, %c72 : index, %c80 : index, %c88 : index, %c96 : index, %c104 : index, %c112 : index, %c120 : index, %c17 : index, %c18 : index, %c19 : index, %c20 : index, %c21 : index, %c22 : index, %c23 : index, %c25 : index, %c26 : index, %c27 : index, %c28 : index, %c29 : index, %c30 : index, %c31 : index, %c33 : index, %c34 : index, %c35 : index, %c36 : index, %c37 : index, %c38 : index, %c39 : index, %c41 : index, %c42 : index, %c43 : index, %c44 : index, %c45 : index, %c46 : index, %c47 : index, %c49 : index, %c50 : index, %c51 : index, %c52 : index, %c53 : index, %c54 : index, %c55 : index, %c57 : index, %c58 : index, %c59 : index, %c60 : index, %c61 : index, %c62 : index, %c63 : index, %c-16 : index, %c-256 : index, %c-8 : index, %cst_1 : vector<2x2xf32>, %memref : memref<5376x2048xf32>, %cst_4 : f32, %memref_0 : memref<2048x5376xf32>, %cst_2 : vector<4x1xf32>, %cst_3 : vector<2x1xf32>, %c2048 : index)
    gpu.dealloc  %memref_0 : memref<2048x5376xf32>
    %alloc = memref.alloc() : memref<5376x5376xf32>
    gpu.memcpy  %alloc, %memref_5 : memref<5376x5376xf32>, memref<5376x5376xf32>
    gpu.dealloc  %memref_5 : memref<5376x5376xf32>

    // ori
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5376x5376xf32>
    gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%c21, %c42, %c1) threads in (%c64, %c2, %c1)  args(%c256 : index, %c0 : index, %c-1 : index, %c32 : index, %c128 : index, %c64 : index, %cst : vector<8xf32>, %alloc : memref<5376x5376xf32>, %c1 : index, %c2 : index, %c3 : index, %c4 : index, %c5 : index, %c6 : index, %c7 : index, %c8 : index, %c9 : index, %c10 : index, %c11 : index, %c12 : index, %c13 : index, %c14 : index, %c15 : index, %c16 : index, %c24 : index, %c40 : index, %c48 : index, %c56 : index, %c72 : index, %c80 : index, %c88 : index, %c96 : index, %c104 : index, %c112 : index, %c120 : index, %c17 : index, %c18 : index, %c19 : index, %c20 : index, %c21 : index, %c22 : index, %c23 : index, %c25 : index, %c26 : index, %c27 : index, %c28 : index, %c29 : index, %c30 : index, %c31 : index, %c33 : index, %c34 : index, %c35 : index, %c36 : index, %c37 : index, %c38 : index, %c39 : index, %c41 : index, %c42 : index, %c43 : index, %c44 : index, %c45 : index, %c46 : index, %c47 : index, %c49 : index, %c50 : index, %c51 : index, %c52 : index, %c53 : index, %c54 : index, %c55 : index, %c57 : index, %c58 : index, %c59 : index, %c60 : index, %c61 : index, %c62 : index, %c63 : index, %c-16 : index, %c-256 : index, %c-8 : index, %cst_0 : vector<2x2xf32>, %arg0 : memref<5376x2048xf32>, %cst_3 : f32, %arg1 : memref<2048x5376xf32>, %cst_1 : vector<4x1xf32>, %cst_2 : vector<2x1xf32>, %c2048 : index)
```
