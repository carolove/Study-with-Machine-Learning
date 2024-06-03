// ../../mlir/llvm-project/build/bin/mlir-opt ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/01-matmul-gpu.mlir  -convert-linalg-to-affine-loops -o ./8-实现案例/2-mlir实现高性能GPU矩阵乘法/source/02-affine-gpu.mlir
module {
    func.func @matmul_linalg(%A: memref<64x32xf32>, %B: memref<32x64xf32>, %C: memref<64x64xf32>) {
        linalg.matmul ins(%A, %B : memref<64x32xf32>, memref<32x64xf32>)
            outs(%C: memref<64x64xf32>)
        return
    }
}
