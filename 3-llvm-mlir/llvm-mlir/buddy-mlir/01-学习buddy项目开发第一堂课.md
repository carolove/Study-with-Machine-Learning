# 01-学习buddy项目开发第一堂课
- 这堂课主要是学习buddy项目的e8aab6c5fcb9489973f0936918f0c62daf6a0c21-ee1b3675468f0aeca0c0b945672596e5594f6046
- 这几个commit开发了一个conversion mlir流程，这个跟mlir tutorial的 《mlir tutorial 学习纪要 03 编写第一个mlir pass》是很类似的，基本上就是手写mlir pass
- 这堂课主要是开发一个手写conversion pass流程

## 最新的match and rewrite 用 strip mining strategy (CB-SM) 策略实现  coefficients broadcasting 算法 应用于conv-2D算子
- 读懂strip mining strategy (CB-SM)技术在高性能计算的应用
- strip mining strategy (CB-SM) 数据带状分解 循环展开策略，是指将大的串行循环，展开为多个小循环，循环展开/数据带状分解的宽度 的标准依据即处理器可并行的数据规模/可寄存器缓存的数据规模
```
我们有可能重复使用 y[i] 和  y[i+1] ，但这需要更复杂的编程。这里的关键是将循环分成若干块。比如

 for (i=0; i<M; i+=2){
   s1 =s2 =0;
   for (j){
     s1 = s1 +a[i][j] * x[j];
     s2 = s2 + a[i+1][j] * x[j];
  }
   y[i] = s1; y[i+1] = s2;
 }
这也被称为「循环展开」（loop unrolling），或「Strip mining」。循环展开的层数由可用寄存器的数量决定。这里的strip=2，并且因为数据规模的问题，j和i是循环倒置的
```
- 代码理解
```
commit page
https://github.com/buddy-compiler/buddy-mlir/commits/main/?after=ee5c0ede479f69e2643b64b46532f72d683467ee+944
LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override { // 这个地方不太理解他是怎么match的怎么就找到这个block来逐行match并替换的。。。
    auto loc = op->getLoc(); 
    auto ctx = op->getContext();
    // Create constant index.
    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);  // 这一行对应的生成后的mlir为 %c0 = arith.constant 0 : index
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    // Get input, kernel and output.
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);
    // Create DimOp.
    Value kernelRow = rewriter.create<memref::DimOp>(loc, kernel, c0);
    Value kernelCol = rewriter.create<memref::DimOp>(loc, kernel, c1);
    Value outputRow = rewriter.create<memref::DimOp>(loc, output, c0);
    Value outputCol = rewriter.create<memref::DimOp>(loc, output, c1);
    // Size of strip mining.
    AffineExpr d0;
    bindDims(ctx, d0);
    AffineMap stripMap = AffineMap::get(1, 0, {d0.ceilDiv(stride)}, ctx);
    SmallVector<Value, 8> lowerBounds(3, c0);
    SmallVector<Value, 8> uperBounds{outputRow, kernelRow, kernelCol};
    SmallVector<int64_t, 8> steps(3, /*Value=*/1);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, uperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          // Create strip mining loop.
          builder.create<AffineForOp>(
              loc, ValueRange{c0}, builder.getDimIdentityMap(),
              ValueRange{outputCol}, stripMap, /*Step=*/1, llvm::None,
              [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                  ValueRange itrArgs) {
                // Vectorize the kernel.
                // Define `*Type`.
                FloatType f32 = mlir::FloatType::getF32(ctx);
                VectorType vectorTy1 = mlir::VectorType::get({1}, f32);
                VectorType vectorTy32 = mlir::VectorType::get({stride}, f32);
                // Broadcast element of the kernel.
                Value kernelValue = builder.create<AffineVectorLoadOp>(
                    loc, vectorTy1, kernel, ValueRange{ivs[1], ivs[2]});
                Value kernelVector =
                    builder.create<BroadcastOp>(loc, vectorTy32, kernelValue);
                // Load input vector from memref.
                AffineExpr m, n, k, j;
                bindDims(ctx, m, n, k, j);
                AffineMap inputVectorMap = AffineMap::get(
                    /*dimCount=*/4, /*symbolCount=*/0, {m + n, k + j * stride},
                    ctx);
                Value inputVector = nestedBuilder.create<AffineVectorLoadOp>(
                    loc, vectorTy32, input, inputVectorMap,
                    ValueRange{ivs[0], ivs[1], ivs[2], iv});
                // Define AffineMap.
                // The `outputVector` and `resultVector` share the same
                // AffineMap.
                AffineExpr x, y;
                bindDims(ctx, x, y);
                AffineMap outputVectorMap = AffineMap::get(
                    /*dimCount=*/2, /*symbolCount=*/0, {x, y * stride}, ctx);
                Value outputVector = nestedBuilder.create<AffineVectorLoadOp>(
                    loc, vectorTy32, output, outputVectorMap,
                    ValueRange{ivs[0], iv});
                // FMA = Fused Multiply + Add
                Value resultVector = nestedBuilder.create<FMAOp>(
                    loc, inputVector, kernelVector, outputVector);
                nestedBuilder.create<AffineVectorStoreOp>(
                    loc, resultVector, output, outputVectorMap,
                    ValueRange{ivs[0], iv});
                nestedBuilder.create<AffineYieldOp>(nestedLoc);
              });
        });
    // Remove the origin convolution operation.
    rewriter.eraseOp(op);
    return success();
  }
```
```
原始的mlir
func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>) outs (%arg2: memref<?x?xf32>)
  return
}

./build/bin/conv-opt examples/conv-opt/conv2d.mlir -conv-vectorization="strip-mining=256"
conversion后的mlir
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 256)>
module  {
  func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %2 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %3 = memref.dim %arg2, %c1 : memref<?x?xf32>
    affine.for %arg3 = #map0(%c0) to #map0(%2) {
      affine.for %arg4 = #map0(%c0) to #map0(%0) {
        affine.for %arg5 = #map0(%c0) to #map0(%1) {
          affine.for %arg6 = #map0(%c0) to #map1(%3) {
            %4 = affine.vector_load %arg1[%arg4, %arg5] : memref<?x?xf32>, vector<1xf32>
            %5 = vector.broadcast %4 : vector<1xf32> to vector<256xf32>
            %6 = affine.vector_load %arg0[%arg3 + %arg4, %arg5 + %arg6 * 256] : memref<?x?xf32>, vector<256xf32>
            %7 = affine.vector_load %arg2[%arg3, %arg6 * 256] : memref<?x?xf32>, vector<256xf32>
            %8 = vector.fma %6, %5, %7 : vector<256xf32>
            affine.vector_store %8, %arg2[%arg3, %arg6 * 256] : memref<?x?xf32>, vector<256xf32>
          }
        }
      }
    }
    return
  }
}

实现这个算法涉及到的 MLIR Dialect 以及 Op 这里列一下：

affine.for ：执行指定次数循环体的操作。
affine.vector_load：从缓冲区切片中返回一个向量 （MLIR MemRef格式）。
affine.vector_store：将一个向量写到缓存区切片中（MLIR MemRef格式）。
vector.broadcast：将标量或向量值广播为 N-维 结果向量。
vector.fma：向量化类型的乘加混合指令。
```
 CB 算法的过程如下图所示
![image](https://github.com/carolove/Study-with-Machine-Learning/assets/834467/6dd7dbb7-4095-42a6-aad5-4ce2641ae01e)
```
注意输入是一个通道数为 1 的图片或者特征图，然后 kernel 的通道数也是1。算法的执行流程大概为：

首先将 kernel 的每个元素使用 vector_load 加载到缓冲区中 并使用 vector.broadcast 广播到 vector1 中。
然后将特征图的元素使用 vector_load 加载到 vector2 中。
第三步将输出特征图的元素使用 vector_load 加载到 vector3 中。
然后使用 vector.fma 将 vector1 和 vector2 相乘并加到 vector3 上。
最后使用 vector_store 将上述结果写回缓冲区中。

```
