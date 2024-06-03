// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/01-matmul-gpu.mlir     --linalg-generalize-named-ops --linalg-bufferize --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-normalize   --affine-loop-tile=tile-sizes=4
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>
module {
  func.func @matmul_linalg(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
    affine.for %arg3 = 0 to 64 step 4 {
      affine.for %arg4 = 0 to 64 step 4 {
        affine.for %arg5 = 0 to 32 step 4 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<64x32xf32>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<32x64xf32>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<64x64xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
