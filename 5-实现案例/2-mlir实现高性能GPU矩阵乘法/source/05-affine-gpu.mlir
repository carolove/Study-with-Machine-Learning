// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/02-linalg-to-affine-with-memref.mlir  -pass-pipeline="builtin.module(func.func(convert-affine-for-to-gpu{gpu-block-dims=1 gpu-thread-dims=1}))" 
module {
  func.func @matmul_linalg(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %0 = arith.subi %c8192, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c8192_1 = arith.constant 8192 : index
    %1 = arith.subi %c8192_1, %c0_0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1_3, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %1, %arg13 = %c1_3, %arg14 = %c1_3) {
      %2 = arith.addi %c0, %arg3 : index
      %3 = arith.addi %c0_0, %arg6 : index
      affine.for %arg15 = 0 to 8192 {
        %4 = memref.load %arg0[%2, %arg15] : memref<8192x8192xf32>
        %5 = memref.load %arg1[%arg15, %3] : memref<8192x8192xf32>
        %6 = memref.load %arg2[%2, %3] : memref<8192x8192xf32>
        %7 = arith.mulf %4, %5 : f32
        %8 = arith.addf %6, %7 : f32
        memref.store %8, %arg2[%2, %3] : memref<8192x8192xf32>
      }
      gpu.terminator
    }
    return
  }
}