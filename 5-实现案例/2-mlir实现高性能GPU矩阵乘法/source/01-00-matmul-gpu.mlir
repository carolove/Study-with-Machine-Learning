module {
    func.func @matmul_linalg(%A: memref<8192x8192xf32>, %B: memref<8192x8192xf32>, %C: memref<8192x8192xf32>) {
        linalg.matmul ins(%A, %B : memref<8192x8192xf32>, memref<8192x8192xf32>)
            outs(%C: memref<8192x8192xf32>)
        return
    }
}
