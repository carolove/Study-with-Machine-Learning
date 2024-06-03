// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/02-linalg-affine.mlir -lower-affine  
module {
  func.func @matmul_linalg(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c8192 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c8192_1 = arith.constant 8192 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c8192_1 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c8192_4 = arith.constant 8192 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %c0_3 to %c8192_4 step %c1_5 {
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