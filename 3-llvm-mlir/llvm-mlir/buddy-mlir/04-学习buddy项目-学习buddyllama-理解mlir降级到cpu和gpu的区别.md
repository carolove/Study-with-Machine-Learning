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
- 
```
class BatchMatMulOptimizePattern : public ConversionPattern {
public:
  explicit BatchMatMulOptimizePattern(MLIRContext *context,int64_t stepPlaceHolderParam): ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,context) { /*linalg::BatchMatmulOp::getOperationName() -> linalg.batch_matmul 匹配重写匹配对象 */
    stepPlaceHolder = stepPlaceHolderParam;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get ElementType of input and output.
    auto A_elementType = A.getType().cast<MemRefType>().getElementType();

    // Some constants.
    const Value c0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(stepPlaceHolder));
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr d2 = rewriter.getAffineDimExpr(2);
    const AffineExpr c0_affine = rewriter.getAffineConstantExpr(0);

    const Value c0_dynamicType = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(A_elementType));
    const Value c0_dynamicType_vec = rewriter.create<vector::SplatOp>(loc, VectorType::get({stepPlaceHolder}, A_elementType), c0_dynamicType);

    // Dims
    Value BATCH = rewriter.create<memref::DimOp>(loc, A, 0); // Batch size
    Value M = rewriter.create<memref::DimOp>(loc, A, 1);     // A row
    Value N = rewriter.create<memref::DimOp>(loc, B, 2);     // B col
    Value K = rewriter.create<memref::DimOp>(loc, B, 1);     // B row

    auto reducedValues = llvm::to_vector<4>(llvm::map_range( ArrayRef<mlir::affine::LoopReduction>{}, [](const mlir::affine::LoopReduction &red) { return red.value; }));

    // Build parallel loop body.
    auto parallelLoop = rewriter.create<affine::AffineParallelOp>(loc, ValueRange(reducedValues).getTypes(), ValueRange{BATCH},ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr( "lowerBoundsGroups", rewriter.getI32TensorAttr(ArrayRef<int32_t>{1})),
            rewriter.getNamedAttr( "upperBoundsGroups", rewriter.getI32TensorAttr(ArrayRef<int32_t>{1})),
            rewriter.getNamedAttr("lowerBoundsMap",AffineMapAttr::get(AffineMap::get(0, 0, {c0_affine},rewriter.getContext()))),
            rewriter.getNamedAttr("upperBoundsMap",AffineMapAttr::get(AffineMap::get(1, 0, {d0}, rewriter.getContext()))),
            rewriter.getNamedAttr("reductions", rewriter.getArrayAttr({})),
            rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(1))});

    auto body = new Block();
    rewriter.setInsertionPointToStart(body);
    body->addArgument(rewriter.getIndexType(), loc);

    Value ivBatch = body->getArguments()[0];

    rewriter.create<affine::AffinePrefetchOp>(loc, A, AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext()),ArrayRef<Value>{ivBatch, c0, c0}, false, 3, true);
    affine::buildAffineLoopNest(rewriter, loc, {c0}, {K}, 1,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value ivB_row = ivRange.front();
          affine::buildAffineLoopNest(builder, loc, {c0}, {M}, 1,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value ivA_row = ivRange.front();
                Value applied_n = builder.create<affine::AffineApplyOp>(loc, AffineMap::get(1, 0, d0.ceilDiv(stepPlaceHolder)),ValueRange{N});
                affine::buildAffineLoopNest(builder, loc, {c0}, {applied_n}, 1,
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      Value ivB_col = ivRange.front();
                      Value a_ele = builder.create<affine::AffineLoadOp>(loc, A, ValueRange{ivBatch, ivA_row, ivB_row});
                      Value a_vec = builder.create<vector::BroadcastOp>(loc,VectorType::get({stepPlaceHolder}, A_elementType),a_ele);
                      Value b_col_cur =builder.create<arith::MulIOp>(loc, ivB_col, step);
                      Value tail_len =builder.create<arith::SubIOp>(loc, N, b_col_cur);
                      Value tail_flag = builder.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, tail_len, step);
                      builder.create<scf::IfOp>(loc, tail_flag,
                          [&](OpBuilder &builder, Location loc) {
                            Value b_vec =builder.create<affine::AffineVectorLoadOp>(loc,VectorType::get({stepPlaceHolder}, A_elementType),B, AffineMap::get(3, 0, {d0, d1, d2 * stepPlaceHolder},rewriter.getContext()),ValueRange{ivBatch, ivB_row, ivB_col});
                            Value c_vec =builder.create<affine::AffineVectorLoadOp>(loc,VectorType::get({stepPlaceHolder},A_elementType),C,AffineMap::get(3, 0, {d0, d1, d2 * stepPlaceHolder},rewriter.getContext()),ValueRange{ivBatch, ivA_row, ivB_col});
                            Value result_vec;
                            if (A_elementType.isIntOrFloat() && 0) { // bug
                              Value add_vec = builder.create<arith::MulIOp>(loc, a_vec, b_vec);
                              result_vec = builder.create<arith::AddIOp>(loc, add_vec, c_vec);
                            } else {
                              result_vec = builder.create<vector::FMAOp>(loc, a_vec, b_vec, c_vec);
                            }
                            builder.create<affine::AffineVectorStoreOp>(loc, result_vec, C,AffineMap::get(3, 0,{d0, d1, d2 * stepPlaceHolder},rewriter.getContext()),ValueRange{ivBatch, ivA_row, ivB_col});
                            builder.create<scf::YieldOp>(loc);
                          },
                          [&](OpBuilder &builder, Location loc) {
                            Value mask_vec =builder.create<vector::CreateMaskOp>(loc,VectorType::get({stepPlaceHolder},rewriter.getI1Type()),ValueRange{tail_len});
                            Value b_col_idx_tail =builder.create<arith::MulIOp>(loc, ivB_col,step);
                            Value b_vec_tail =builder.create<vector::MaskedLoadOp>(loc,VectorType::get({stepPlaceHolder},A_elementType),B,ValueRange{ivBatch, ivB_row,b_col_idx_tail},mask_vec, c0_dynamicType_vec);
                            Value c_vec_tail =builder.create<vector::MaskedLoadOp>(loc,VectorType::get({stepPlaceHolder},A_elementType),C,ValueRange{ivBatch, ivA_row,b_col_idx_tail},mask_vec, c0_dynamicType_vec);
                            Value result_vec_tail;
                            if (A_elementType.isIntOrFloat() && 0) { // bug
                              Value add_vec = builder.create<arith::MulIOp>(loc, a_vec, b_vec_tail);
                              result_vec_tail = builder.create<arith::AddIOp>(loc, add_vec, c_vec_tail);
                            } else {
                              result_vec_tail = builder.create<vector::FMAOp>(loc, a_vec, b_vec_tail, c_vec_tail);
                            }
                            builder.create<vector::MaskedStoreOp>(loc, C,ValueRange{ivBatch, ivA_row, b_col_idx_tail},mask_vec, result_vec_tail);
                            builder.create<scf::YieldOp>(loc);
                          });
                    });
              });
        });

    rewriter.create<affine::AffineYieldOp>(loc);
    parallelLoop.getRegion().push_back(body);
    rewriter.setInsertionPointAfter(parallelLoop);

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t stepPlaceHolder;
};
} // end anonymous namespace
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
