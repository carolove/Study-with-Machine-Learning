# mlir pipeline学习
- 这是将matmul lowering到nvidia gpu的整个过程
```
../../build/bin/buddy-opt matmul.mlir -transform-preload-library="transform-library-paths=transform.mlir" -transform-interpreter="entry-point=codegen" -o matmul-transform.mlir 
../../build/bin/buddy-opt matmul-transform.mlir --pass-pipeline='builtin.module(func.func(nvgpu-optimize-shared-memory))' -o matmul-shared-memory.mlir 
../../build/bin/buddy-opt matmul-shared-memory.mlir  -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize  -o matmul-lowering.mlir 
../../build/bin/buddy-opt matmul-lowering.mlir -gpu-launch-sink-index-computations -canonicalize -legalize-shmem-outlining -canonicalize -o matmul-gpu-launch.mlir 
../../build/bin/buddy-opt matmul-gpu-launch.mlir -convert-memcpy-to-gpu -gpu-async-region -canonicalize -o matmul-gpu-async.mlir 
../../build/bin/buddy-opt matmul-gpu-async.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm --convert-vector-to-llvm -convert-gpu-to-nvvm='has-redux=1'  -o matmul-nvvm.mlir
../../build/bin/buddy-opt matmul-nvvm.mlir -llvm-request-c-wrappers -canonicalize -cse -sccp  -o matmul-c-wrapper.mlir 
../../llvm/build/bin/mlir-opt matmul-c-wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o matmul-cubin.mlir 
```
