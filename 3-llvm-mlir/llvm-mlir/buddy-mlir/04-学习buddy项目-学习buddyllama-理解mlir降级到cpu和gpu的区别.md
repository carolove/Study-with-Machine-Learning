# 理解mlir降级到cpu和gpu的区别
- 在第三篇中我们已经学习了DynamoCompiler，理解了通过torch 的compile jit技术，可以将model 转化为 fx graph/aten ir，然后进一步convert到自定义mlir ir表示层（主要是linag dialect以及tosa dialect）上
- 这一章主要学习在获得mlir ir的情况下，如何生成target（比如cpu、gpu）设备运行码的过程和逻辑
- 其次，还需要通过不同的pr，主要是[cpu pr](https://github.com/buddy-compiler/buddy-mlir/pull/216)与[gpu pr](https://github.com/buddy-compiler/buddy-mlir/pull/285)，通过对两个pr的学习理解，弄清楚lowering到异构设备的流程和逻辑，以及主要的程序设计、算法设计

## 对于理解mlir 的重要的pr
### [broadcast BatchMatMul optimization pass, 广播批量矩阵乘法优化算法](https://github.com/buddy-compiler/buddy-mlir/pull/187)
- 理论分析
```
Tensor, tiling 以及 fusion
tosa dialect 可以转换成 linalg dialect。 这种转换会保持在张量这一抽象层级，所以其目的并非递降，而是为接下来的转换做准备。 mhlo.dot_general op 和 tosa.matmul op 都可以表示 batch matmul，那么 linalg.batch_matmul op 的意义何在呢？ 因为隐性嵌套循环，tiling 和 fusion 这些对 tile-based 架构非常重要的转换在 linalg.batch_matmul op 上进行更加方便—我们只需要创建显式的嵌套循环，把之前的 linalg op 转移到其内并且缩小 linalg op 操作的范围到一个 slice 就可以了。

比如下面的 tosa.conv2d op：

%0 = "tosa.conv2d"(%input, %filter, %bias)
       {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]}
     : (tensor<1x225x225x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>)
     -> tensor<1x112x112x32xf32>
转换成 linalg op 并进行 tiling 和 fusion 之后：

%0 = scf.for %iv0 = ... to ... step ... iter_args(...) -> (tensor<1x112x112x32xf32>) {
  %1 = scf.for ... {
    %input_slice = tensor.extract_slice ...
    %filter_slice = tensor.extract_slice ...
    %bias_slice = tensor.extract_slice ...
    %conv = linalg.conv_2d_nhwc_hwcf {...} ins(%input_slice, %filter_slice) ...
    %generic = linalg.generic ins(%conv, %bias_slice} ... {
      %add = arith.addf ...
      linalg.yield %add ...
    }
    scf.yield %generic
  }
  scf.yield %1
}
在嵌套循环之内，我们依然维持着 linalg named op 的形态，以便于进一步的 tiling 和 fusion， 或者进行其他的模式匹配和转换。

```
- linalg.matmul 重写逻辑
```
LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/, ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Retrieve input tensors A, B, and C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Acquire the element type of input tensors.
    Type elementType = A.getType().cast<MemRefType>().getElementType();

    // Define constants.
    const Value zeroIndex = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)); // %c0 = arith.constant 0 : index  
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr zeroAffine = rewriter.getAffineConstantExpr(0);

    const Value zeroElementType = rewriter.create<arith::ConstantOp>( loc, rewriter.getZeroAttr(elementType)); //  %cst = arith.constant 0.000000e+00 : f32
    const Value zeroElementTypeVec = rewriter.create<vector::SplatOp>(loc, VectorType::get({affineVectorSize}, elementType), zeroElementType); //  %3 = vector.splat %cst : vector<64xf32>

    // Get dimensions of input tensors.
    // %c0_0 = arith.constant 0 : index
    // %dim = memref.dim %0, %c0_0 : memref<2x2x3xf32>
    // %c1 = arith.constant 1 : index
    // %dim_1 = memref.dim %0, %c1 : memref<2x2x3xf32>
    // %c2 = arith.constant 2 : index
    // %dim_2 = memref.dim %1, %c2 : memref<2x3x4xf32>
    // %c1_3 = arith.constant 1 : index
    // %dim_4 = memref.dim %1, %c1_3 : memref<2x3x4xf32>
    Value batch = rewriter.create<memref::DimOp>(loc, A, 0);
    Value aRow = rewriter.create<memref::DimOp>(loc, A, 1);
    Value bCol = rewriter.create<memref::DimOp>(loc, B, 2);
    Value bRow = rewriter.create<memref::DimOp>(loc, B, 1);

    // Calculate the length of the tail, which might not fit in a vector.
    Value tailLength = rewriter.create<affine::AffineApplyOp>( loc, AffineMap::get(1, 0, d0 % affineVectorSize), ValueRange{bCol}); //  %4 = affine.apply #map(%dim_2)

    // Generate a mask vector based on the tail length.
    Value maskVector = rewriter.create<vector::CreateMaskOp>(loc, VectorType::get({affineVectorSize}, rewriter.getI1Type()),  ValueRange{tailLength}); // %5 = vector.create_mask %4 : vector<64xi1>

    SmallVector<Value, 4U> reducedValues = llvm::to_vector<4>( llvm::map_range(ArrayRef<LoopReduction>{},     [](const LoopReduction &red) { return red.value; }));

    // Apply the column of matrix B.
    Value appliedColOfB = rewriter.create<affine::AffineApplyOp>( loc, AffineMap::get(1, 0, d0.ceilDiv(affineVectorSize)),ValueRange{bCol}); // %6 = affine.apply #map1(%dim_2)

    // Create the primary parallel batch level loop.
    //  affine.parallel (%arg0) = (0) to (%dim) {
    AffineParallelOp parallelBatchLoop = rewriter.create<affine::AffineParallelOp>(loc, ValueRange(reducedValues).getTypes(), ValueRange{batch},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("lowerBoundsGroups",rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr("upperBoundsGroups", rewriter.getI32TensorAttr({1})),
                rewriter.getNamedAttr( "lowerBoundsMap", AffineMapAttr::get(AffineMap::get(0, 0, {zeroAffine},rewriter.getContext()))),
                rewriter.getNamedAttr("upperBoundsMap", AffineMapAttr::get(AffineMap::get(1, 0, {d0}, rewriter.getContext()))),
                rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
                rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1}))});

    // Create the loop body for the parallel loop.
    Block *loopBody = new Block();
    rewriter.setInsertionPointToStart(loopBody);
    loopBody->addArgument(rewriter.getIndexType(), loc);
    Value loopVarBatchIdx = loopBody->getArguments()[0];

    // Prefetching data from tensor 'A' for better cache utilization.
    rewriter.create<affine::AffinePrefetchOp>(loc, A, AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext()),ArrayRef<Value>{loopVarBatchIdx, aRow, bRow}, false, 3, true); // affine.prefetch %0[%arg0, %dim_1, %dim_4], read, locality<3>, data : memref<2x2x3xf32>

    affine::buildAffineLoopNest( rewriter, loc, {zeroIndex}, {appliedColOfB}, 1,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value loopVarColOfB = ivRange.front();

          // Compile time branch detection.
          if (C.getType().cast<MemRefType>().isDynamicDim(2) or C.getType().cast<MemRefType>().getDimSize(2) % affineVectorSize !=0) {
          // affine.if #set(%arg0)[%dim_2] {
          //   affine.for %arg2 = #map2(%c0) to #map2(%dim_4) {
          //     %7 = affine.vector_load %1[%arg0, %arg2, %arg1 * 64] : memref<2x3x4xf32>, vector<64xf32>
          //     affine.for %arg3 = #map2(%c0) to #map2(%dim_1) {
          //       %8 = memref.load %0[%arg0, %arg3, %arg2] : memref<2x2x3xf32>
          //       %9 = vector.broadcast %8 : f32 to vector<64xf32>
          //       %10 = affine.vector_load %2[%arg0, %arg3, %arg1 * 64] : memref<2x2x4xf32>, vector<64xf32>
          //       %11 = vector.fma %9, %7, %10 : vector<64xf32>
          //       affine.vector_store %11, %2[%arg0, %arg3, %arg1 * 64] : memref<2x2x4xf32>, vector<64xf32>
          //     }
          //   }
          // } 
            // Depending on the position, use either full vectors or tail
            // vectors.
            affine::AffineIfOp branchingOp = builder.create<affine::AffineIfOp>( loc, IntegerSet::get( 1, 1, {d0 * -affineVectorSize + s0 - affineVectorSize}, {false}),  ValueRange{loopVarBatchIdx, bCol}, true);

            // Branch handling full vector operations.
            OpBuilder trueBranchBuilder = branchingOp.getThenBodyBuilder();
            affine::buildAffineLoopNest( trueBranchBuilder, loc, {zeroIndex}, {bRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                  Value loopVarRowOfB = ivRange.front();
                  Value bVec = builder.create<affine::AffineVectorLoadOp>(loc, VectorType::get({affineVectorSize}, elementType), B,AffineMap::get(3, 0, {d0, d1, d2 * affineVectorSize}, rewriter.getContext()),ValueRange{loopVarBatchIdx, loopVarRowOfB, loopVarColOfB});
                  affine::buildAffineLoopNest(builder, loc, {zeroIndex}, {aRow}, 1,
                      [&](OpBuilder &builder, Location loc,ValueRange ivRange) {
                        Value loopVarRowOfA = ivRange.front();
                        Value aElement = builder.create<memref::LoadOp>(loc, A,ValueRange{loopVarBatchIdx, loopVarRowOfA,loopVarRowOfB});
                        Value aVec = builder.create<vector::BroadcastOp>( loc,VectorType::get({affineVectorSize}, elementType),aElement);
                        Value cVec = builder.create<affine::AffineVectorLoadOp>( loc, VectorType::get({affineVectorSize}, elementType), C, AffineMap::get(3, 0, {d0, d1, d2 * affineVectorSize},   builder.getContext()), ValueRange{loopVarBatchIdx, loopVarRowOfA,loopVarColOfB});
                        Value computedVec;

                        // Compute the result vector either through integer
                        // multiplication and addition or fused multiply-add
                        // based on the element type.
                        if (isa<IntegerType>(elementType)) {
                             Value mulVec = builder.create<arith::MulIOp>(loc, aVec, bVec);
                          computedVec = builder.create<arith::AddIOp>(loc, mulVec, cVec);
                        } else {
                          computedVec = builder.create<vector::FMAOp>(loc, aVec, bVec, cVec);
                        }
                        builder.create<affine::AffineVectorStoreOp>(loc, computedVec, C, AffineMap::get(3, 0,{d0, d1, d2 * affineVectorSize},builder.getContext()),  ValueRange{loopVarBatchIdx, loopVarRowOfA,  loopVarColOfB});
                      });
                });

            // Branch handling operations on the tail.
            OpBuilder falseBranchBuilder = branchingOp.getElseBodyBuilder();
            affine::buildAffineLoopNest( falseBranchBuilder, loc, {zeroIndex}, {bRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                  Value loopVarRowOfB = ivRange.front();
                  Value tailIdxColOfB = builder.create<affine::AffineApplyOp>(  loc, AffineMap::get(1, 0, d0 * affineVectorSize), ValueRange{loopVarColOfB});
                  Value bVec = builder.create<vector::MaskedLoadOp>(loc, VectorType::get({affineVectorSize}, elementType), B, ValueRange{loopVarBatchIdx, loopVarRowOfB, tailIdxColOfB}, maskVector, zeroElementTypeVec);
                  affine::buildAffineLoopNest( builder, loc, {zeroIndex}, {aRow}, 1,
                      [&](OpBuilder &builder, Location loc,ValueRange ivRange) {
                        Value loopVarRowOfA = ivRange.front();
                        Value aElement = builder.create<memref::LoadOp>( loc, A, ValueRange{loopVarBatchIdx, loopVarRowOfA,          loopVarRowOfB});
                        Value aVec = builder.create<vector::BroadcastOp>( loc, VectorType::get({affineVectorSize}, elementType), aElement);
                        Value cVec = builder.create<vector::MaskedLoadOp>( loc, VectorType::get({affineVectorSize}, elementType), C,ValueRange{loopVarBatchIdx, loopVarRowOfA,            tailIdxColOfB},   maskVector, zeroElementTypeVec);
                        Value computedVec;

                        // Compute the result vector either through integer
                        // multiplication and addition or fused multiply-add
                        // based on the element type.
                        if (isa<IntegerType>(elementType)) {
                          Value mulVec =  builder.create<arith::MulIOp>(loc, aVec, bVec);
                          computedVec =  builder.create<arith::AddIOp>(loc, mulVec, cVec);
                        } else {
                          computedVec = builder.create<vector::FMAOp>(  loc, aVec, bVec, cVec);
                        }
                        builder.create<vector::MaskedStoreOp>(  loc, C,  ValueRange{loopVarBatchIdx, loopVarRowOfA,tailIdxColOfB},  maskVector, computedVec);
                      });
                });
          } else {
            // else {
            // affine.for %arg2 = #map2(%c0) to #map2(%dim_4) {
            // %7 = affine.apply #map3(%arg1)
            // %8 = vector.maskedload %1[%arg0, %arg2, %7], %5, %3 : memref<2x3x4xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            // affine.for %arg3 = #map2(%c0) to #map2(%dim_1) {
            //   %9 = memref.load %0[%arg0, %arg3, %arg2] : memref<2x2x3xf32>
            //   %10 = vector.broadcast %9 : f32 to vector<64xf32>
            //   %11 = vector.maskedload %2[%arg0, %arg3, %7], %5, %3 : memref<2x2x4xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            //   %12 = vector.fma %10, %8, %11 : vector<64xf32>
            //   vector.maskedstore %2[%arg0, %arg3, %7], %5, %12 : memref<2x2x4xf32>, vector<64xi1>, vector<64xf32>
            // }
            // }
            // }
            // }
            affine::buildAffineLoopNest( builder, loc, {zeroIndex}, {bRow}, 1,
                [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                  Value loopVarRowOfB = ivRange.front();
                  Value bVec = builder.create<affine::AffineVectorLoadOp>(loc, VectorType::get({affineVectorSize}, elementType), B, AffineMap::get(3, 0, {d0, d1, d2 * affineVectorSize}, rewriter.getContext()), ValueRange{loopVarBatchIdx, loopVarRowOfB, loopVarColOfB});
                  affine::buildAffineLoopNest(builder, loc, {zeroIndex}, {aRow}, 1,
                      [&](OpBuilder &builder, Location loc,ValueRange ivRange) {
                        Value loopVarRowOfA = ivRange.front();
                        Value aElement = builder.create<memref::LoadOp>(loc, A,ValueRange{loopVarBatchIdx, loopVarRowOfA,loopVarRowOfB});
                        Value aVec = builder.create<vector::BroadcastOp>(loc,VectorType::get({affineVectorSize}, elementType),aElement);
                        Value cVec = builder.create<affine::AffineVectorLoadOp>(loc,VectorType::get({affineVectorSize}, elementType), C,AffineMap::get(3, 0,{d0, d1, d2 * affineVectorSize},builder.getContext()),ValueRange{loopVarBatchIdx, loopVarRowOfA,loopVarColOfB});
                        Value computedVec;

                        // Compute the result vector either through integer
                        // multiplication and addition or fused multiply-add
                        // based on the element type.
                        if (isa<IntegerType>(elementType)) {
                          Value mulVec =builder.create<arith::MulIOp>(loc, aVec, bVec);
                          computedVec =builder.create<arith::AddIOp>(loc, mulVec, cVec);
                        } else {
                          computedVec = builder.create<vector::FMAOp>(loc, aVec, bVec, cVec);
                        }
                        builder.create<affine::AffineVectorStoreOp>(loc, computedVec, C,AffineMap::get(3, 0,{d0, d1, d2 * affineVectorSize},builder.getContext()),ValueRange{loopVarBatchIdx, loopVarRowOfA,loopVarColOfB});
                      });
                });
          }
        });

    rewriter.create<affine::AffineYieldOp>(loc);

    // Finalize the loop and erase the original operation.
    parallelBatchLoop.getRegion().push_back(loopBody);
    rewriter.setInsertionPointAfter(parallelBatchLoop);

    rewriter.eraseOp(op);
    return success();
  }

```
mlir test filecheck
```
// RUN: buddy-opt -batchmatmul-optimize -verify-diagnostics -expand-strided-metadata -lower-affine -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-linalg-to-llvm -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts %s | mlir-cpu-runner -O0 -e buddy_batchmatmul_f32  -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

memref.global "private" @A : memref<2x2x3xf32> = dense<[[[9., 4., 6.],[2., 4., 0.]],[[6., 3., 3.],[0., 4., 7.]]]>
memref.global "private" @B : memref<2x3x4xf32> = dense<[[[1., 3., 8., 0.],[1., 8., 8., 7.], [6., 9., 7., 9.]],[[3., 8., 6., 8.],[2., 7., 0., 6.],[0., 4., 0., 4.]]]>
memref.global "private" @C : memref<2x2x4xf32> = dense<[[[ 49., 113., 146.,  82.],[  6.,  38.,  48.,  28.]],[[ 24.,  81.,  36.,  78.],[  8.,  56.,   0.,  52.]]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_batchmatmul_f32() -> f32{
  %a = memref.get_global @A : memref<2x2x3xf32>
  %b = memref.get_global @B : memref<2x3x4xf32>
  %c = memref.get_global @C : memref<2x2x4xf32>

  linalg.batch_matmul 
      ins(%a, %b: memref<2x2x3xf32>, memref<2x3x4xf32>)
      outs(%c: memref<2x2x4xf32>)
  %printed_c = memref.cast %c : memref<2x2x4xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[2, 2, 4\] strides = \[8, 4, 1\] data =}}
  // CHECK{LITERAL}: [[[98,    226,    292,    164], 
  // CHECK{LITERAL}:   [12,    76,    96,    56]], 
  // CHECK{LITERAL}:  [[48,    162,    72,    156], 
  // CHECK{LITERAL}:   [16,    112,    0,    104]]]
  %zero = arith.constant 0.0 :f32
  return %zero :f32
}
```
output 其中 linalg.batch_matmul 替换为了
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %3 = vector.splat %cst : vector<64xf32>
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %0, %c0_0 : memref<2x2x3xf32>
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %0, %c1 : memref<2x2x3xf32>
    %c2 = arith.constant 2 : index
    %dim_2 = memref.dim %1, %c2 : memref<2x3x4xf32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %1, %c1_3 : memref<2x3x4xf32>
    %4 = affine.apply #map(%dim_2)
    %5 = vector.create_mask %4 : vector<64xi1>
    %6 = affine.apply #map1(%dim_2)
    affine.parallel (%arg0) = (0) to (%dim) {
      affine.prefetch %0[%arg0, %dim_1, %dim_4], read, locality<3>, data : memref<2x2x3xf32>
      affine.for %arg1 = #map2(%c0) to #map2(%6) {
        affine.if #set(%arg0)[%dim_2] {
          affine.for %arg2 = #map2(%c0) to #map2(%dim_4) {
            %7 = affine.vector_load %1[%arg0, %arg2, %arg1 * 64] : memref<2x3x4xf32>, vector<64xf32>
            affine.for %arg3 = #map2(%c0) to #map2(%dim_1) {
              %8 = memref.load %0[%arg0, %arg3, %arg2] : memref<2x2x3xf32>
              %9 = vector.broadcast %8 : f32 to vector<64xf32>
              %10 = affine.vector_load %2[%arg0, %arg3, %arg1 * 64] : memref<2x2x4xf32>, vector<64xf32>
              %11 = vector.fma %9, %7, %10 : vector<64xf32>
              affine.vector_store %11, %2[%arg0, %arg3, %arg1 * 64] : memref<2x2x4xf32>, vector<64xf32>
            }
          }
        } else {
          affine.for %arg2 = #map2(%c0) to #map2(%dim_4) {
            %7 = affine.apply #map3(%arg1)
            %8 = vector.maskedload %1[%arg0, %arg2, %7], %5, %3 : memref<2x3x4xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
            affine.for %arg3 = #map2(%c0) to #map2(%dim_1) {
              %9 = memref.load %0[%arg0, %arg3, %arg2] : memref<2x2x3xf32>
              %10 = vector.broadcast %9 : f32 to vector<64xf32>
              %11 = vector.maskedload %2[%arg0, %arg3, %7], %5, %3 : memref<2x2x4xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
              %12 = vector.fma %10, %8, %11 : vector<64xf32>
              vector.maskedstore %2[%arg0, %arg3, %7], %5, %12 : memref<2x2x4xf32>, vector<64xi1>, vector<64xf32>
            }
          }
        }
      }
    }
