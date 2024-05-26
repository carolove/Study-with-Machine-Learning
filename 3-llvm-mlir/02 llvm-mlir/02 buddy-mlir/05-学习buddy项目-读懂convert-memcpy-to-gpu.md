# 学习buddy项目-读懂convert-memcpy-to-gpu
## 原理
```
其实就是将gpu.launch_func 涉及的矩阵入参，将memref替换为gpu.alloc&gpu.memcpy&gpu.dealloc
```
## 核心代码
### (一) 修改入参
```
auto funcOp = getOperation();
std::set<gpu::AllocOp *> unDeallocatedOperations;
  OpBuilder builder(funcOp->getContext());
  // Copy all function arguments to gpu, needs deallocation
  if (processArgs) {
    builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    unsigned numArgs = funcOp.getNumArguments();
    for (unsigned i = 0; i < numArgs; ++i) {
      BlockArgument arg = funcOp.getArgument(i);
      // Create a gpu.alloc op, then copy memory to it
      // TODO: Move this out of operation, make the copy process async
      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          builder.getUnknownLoc(), TypeRange({memrefType}), ValueRange({}));
      unDeallocatedOperations.insert(&gpuAllocOp);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          gpuAllocOp.getLoc(), TypeRange(), ValueRange(),
          gpuAllocOp.getResult(0), arg);
      // Replace all users with GPU memory
      auto users = arg.getUsers();
      std::vector<Operation *> usersVec(users.begin(), users.end());
      for (auto user : usersVec) {
        // Don't replace memcpy's operand
        if (isa<gpu::MemcpyOp>(user))
          continue;
        for (size_t j = 0; j < user->getNumOperands(); j++) {
          if (user->getOperand(j) == arg) {
            user->setOperand(j, gpuAllocOp.getResult(0));
          }
        }
      }
    }
  }
```
**这部分相当于将入参修改为gpu memory，基本上是insert操作，没有op erase**
```
    %memref = gpu.alloc  () : memref<5376x2048xf32>
    gpu.memcpy  %memref, %arg0 : memref<5376x2048xf32>, memref<5376x2048xf32>
    %memref_0 = gpu.alloc  () : memref<2048x5376xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<2048x5376xf32>, memref<2048x5376xf32>
```

### (二) 修改代码块中的memref
- **这部分是 memref.alloc()  替换为gpu alloc**
```
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    // Replace all allocations with GPU.alloc
    if (auto allocOp = dyn_cast<memref::AllocOp>(nestedOp)) {  
      // Rewrite this allocOp to gpu.alloc, change for all users
      builder.setInsertionPointAfter(allocOp);
      auto result = allocOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto memorySpace = memrefType.getMemorySpace();

      // Filter operations.
      if (memorySpace) {
        if (auto intMemorySpace = llvm::dyn_cast<IntegerAttr>(memorySpace)) {
          if (intMemorySpace.getInt() != 0) {
            return WalkResult::advance();
          }
        }
        else if (auto gpuMemorySpace = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)){
          if (gpuMemorySpace.getValue()!=gpu::AddressSpace::Global) {
            return WalkResult::advance();
          }
        }
        else return WalkResult::advance();
      }

      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          allocOp->getLoc(), TypeRange({memrefType}), ValueRange({}));
      auto users = result.getUsers();
      std::vector<Operation *> usersVec(users.begin(), users.end());
      for (auto user : usersVec) {
        for (size_t j = 0; j < user->getNumOperands(); j++) {
          // Only the return value will not have dealloc op
          if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
            builder.setInsertionPointAfter(deallocOp);
            auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
                deallocOp->getLoc(), TypeRange(), ValueRange(),
                gpuAllocOp.getResult(0));
            deallocOp->erase();
          } else if (user->getOperand(j) == result) {
            user->setOperand(j, gpuAllocOp.getResult(0));
          }
        }
      }
      allocOp->erase();  // 需要删除原来的memref.alloc() 
    }
    // Replace all memory.copy operations with gpu.memcpy
```
 - **这次演示中应该没有cpy,在这个项目中其实是可以删除的，并没有用到**
```
    else if (auto copyOp = dyn_cast<memref::CopyOp>(nestedOp)) {
      auto src = copyOp.getOperand(0);
      auto dst = copyOp.getOperand(1);
      // Notice: GPU.memcpy has a different src dst order
      builder.setInsertionPointAfter(copyOp);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          copyOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      {
        auto users = src.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, gpuMemcpyOp.getOperand(1));
            }
          }
        }
      }
      {
        auto users = dst.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, gpuMemcpyOp.getOperand(0));
            }
          }
        }
      }
      copyOp->erase();
    }
```
- **这部分应该也是没有的，不需要将memory cpy去gpu，已经在入参操作完成了**
```
    // Allocate space on GPU and copy global memrefs to GPU, needs deallocation
    else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(nestedOp)) {
      builder.setInsertionPointAfter(getGlobalOp);
      auto result = getGlobalOp->getResult(0);
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      auto gpuAllocOp = builder.create<gpu::AllocOp>(
          getGlobalOp->getLoc(), TypeRange({memrefType}), ValueRange({}));
      unDeallocatedOperations.insert(&gpuAllocOp);
      auto src = result;
      auto dst = gpuAllocOp->getResult(0);
      auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
          gpuAllocOp->getLoc(), TypeRange(), ValueRange(), dst, src);
      {
        auto users = src.getUsers();
        std::vector<Operation *> usersVec(users.begin(), users.end());
        for (auto user : usersVec) {
          if (isa<gpu::MemcpyOp>(user))
            continue;
          // TODO: replace with src.replaceAllUsesExcept()
          for (size_t j = 0; j < user->getNumOperands(); j++) {
            if (user->getOperand(j) == src) {
              user->setOperand(j, dst);
            }
          }
        }
      }
    }
```
- **这是将最终的return 替换为**
```
%alloc = memref.alloc() : memref<5376x5376xf32>
gpu.memcpy  %alloc, %memref_5 : memref<5376x5376xf32>, memref<5376x5376xf32>
gpu.dealloc  %memref_5 : memref<5376x5376xf32>
return %alloc : memref<5376x5376xf32>
```
```
    // Copy data back to CPU, deallocate GPU, then return
    else if (auto returnOp = dyn_cast<func::ReturnOp>(nestedOp)) {
      builder.setInsertionPoint(returnOp);

      for (auto *gpuAllocOp : unDeallocatedOperations) {
        auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
            builder.getUnknownLoc(), TypeRange(), ValueRange(),
            gpuAllocOp->getResult(0));
      }
      builder.setInsertionPoint(returnOp);
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto val = returnOp->getOperand(i);
        auto memRefType = dyn_cast<MemRefType>(val.getType());
        auto allocOp = builder.create<memref::AllocOp>(builder.getUnknownLoc(),
                                                       memRefType);
        auto gpuMemcpyOp = builder.create<gpu::MemcpyOp>(
            allocOp.getLoc(), TypeRange(), ValueRange(), allocOp->getResult(0),
            val);
        auto gpuDeallocOp = builder.create<gpu::DeallocOp>(
            gpuMemcpyOp->getLoc(), TypeRange(), ValueRange(), val);
        returnOp->setOperand(i, allocOp->getResult(0));
      }
    }
    return WalkResult::advance();
  })
```
```
    从
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5376x5376xf32>
    gpu.launch_func  @matmul_kernel::@matmul_kernel(
    return %alloc : memref<5376x5376xf32>
    替换为
    %memref_5 = gpu.alloc  () : memref<5376x5376xf32>
    gpu.dealloc  %memref_0 : memref<2048x5376xf32>
    gpu.launch_func  @matmul_kernel::@matmul_kernel(
    %alloc = memref.alloc() : memref<5376x5376xf32>
    gpu.memcpy  %alloc, %memref_5 : memref<5376x5376xf32>, memref<5376x5376xf32>
    gpu.dealloc  %memref_5 : memref<5376x5376xf32>
    return %alloc : memref<5376x5376xf32>
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
