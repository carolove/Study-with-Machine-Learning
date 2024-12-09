// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/01-00-matmul-gpu.mlir --linalg-generalize-named-ops                                 
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_linalg(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<8192x8192xf32>, memref<8192x8192xf32>) outs(%arg2 : memref<8192x8192xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
}