# 理解mlir降级到cpu和gpu的区别
- 在第三篇中我们已经学习了DynamoCompiler，理解了通过torch 的compile jit技术，可以将model 转化为 fx graph/aten ir，然后进一步convert到自定义mlir ir表示层（主要是linag dialect以及tosa dialect）上
- 这一章主要学习在获得mlir ir的情况下，如何生成target（比如cpu、gpu）设备运行码的过程和逻辑
- 其次，还需要通过不同的pr，主要是[cpu pr](https://github.com/buddy-compiler/buddy-mlir/pull/216)与[gpu pr](https://github.com/buddy-compiler/buddy-mlir/pull/285)，通过对两个pr的学习理解，弄清楚lowering到异构设备的流程和逻辑，以及主要的程序设计、算法设计

 ## 对于理解mlir 的重要的pr
 - [broadcast BatchMatMul optimization pass, 广播批量矩阵乘法优化算法](https://github.com/buddy-compiler/buddy-mlir/pull/187)
```
class BatchMatMulOptimizePattern : public ConversionPattern {
public:
  explicit BatchMatMulOptimizePattern(MLIRContext *context,int64_t stepPlaceHolderParam): ConversionPattern(linalg::BatchMatmulOp::getOperationName(), 1,context) {
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
