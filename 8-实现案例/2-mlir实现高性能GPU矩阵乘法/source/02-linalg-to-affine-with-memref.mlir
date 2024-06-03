// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/01-00-matmul-gpu.mlir --convert-linalg-to-affine-loops 
module {
  func.func @matmul_linalg(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) {
    affine.for %arg3 = 0 to 8192 {
      affine.for %arg4 = 0 to 8192 {
        affine.for %arg5 = 0 to 8192 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8192x8192xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<8192x8192xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8192x8192xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8192x8192xf32>
        }
      }
    }
    return
  }
}

