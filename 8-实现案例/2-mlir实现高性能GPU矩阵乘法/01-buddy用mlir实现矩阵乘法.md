# buddy compile 提供的矩阵乘法mlir生成
- 相关pr [Add single operator optimize operations.](https://github.com/buddy-compiler/buddy-mlir/pull/290)

## 实现的其过程的主要路径
- 从pass lowering命令执行过程来看，如下
```
../../build/bin/buddy-opt matmul.mlir -transform-preload-library="transform-library-paths=transform.mlir" -transform-interpreter="entry-point=codegen" -o ex_source/01-matmul-transform.mlir 
../../llvm/build/bin/mlir-opt ex_source/01-matmul-transform.mlir --pass-pipeline='builtin.module(func.func(nvgpu-optimize-shared-memory))' -o ex_source/02-matmul-shared-memory.mlir 
../../llvm/build/bin/mlir-opt ex_source/02-matmul-shared-memory.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o ex_source/03-matmul-lowering.mlir 
../../llvm/build/bin/mlir-opt ex_source/03-matmul-lowering.mlir -gpu-launch-sink-index-computations -canonicalize -o ex_source/04-00-matmul-gpu-launch.mlir
../../build/bin/buddy-opt ex_source/04-00-matmul-gpu-launch.mlir -legalize-shmem-outlining -canonicalize -o ex_source/04-01-matmul-gpu-shmem-outlining.mlir
../../build/bin/buddy-opt ex_source/04-01-matmul-gpu-shmem-outlining.mlir -convert-memcpy-to-gpu -o ex_source/05-00-matmul-memcpy-gpu.mlir
../../llvm/build/bin/mlir-opt ex_source/05-00-matmul-memcpy-gpu.mlir -gpu-async-region -canonicalize -o ex_source/05-01-matmul-gpu-async.mlir
../../llvm/build/bin/mlir-opt ex_source/05-01-matmul-gpu-async.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm --convert-vector-to-llvm -convert-gpu-to-nvvm='has-redux=1' -o ex_source/06-matmul-nvvm.mlir
../../llvm/build/bin/mlir-opt ex_source/06-matmul-nvvm.mlir -llvm-request-c-wrappers -canonicalize -cse -sccp -o ex_source/07-matmul-c-wrapper.mlir
../../llvm/build/bin/mlir-opt ex_source/07-matmul-c-wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o ex_source/08-matmul-cubin.mlir
```
