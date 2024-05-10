# 读懂TorchDynamoCompiler
- 目标 读懂TorchDynamoCompiler，了解从llm 模型导入到mlir ir生成的逻辑，知其然知其所以然

## TorchDynamoCompiler 核心算法以及数据结构
- commit Add tosa operators
- 整个Tensor Operator Set Architecture (TOSA) op在mlir dialect定义的所有op为 https://mlir.llvm.org/docs/Dialects/TOSA/
- 这个项目需要的tosa的wrapper为
```
frontend/Python/ops/tosa.py中注册的

ops_registry = {
    "AddOp": add_op,
    "MulOp": mul_op,
    "SubOp": sub_op,
    "SumDimOp": sum_op,
    "TanhOp": tanh_op,
    "AmaxOp": amax_op,
    "RsqrtOp": rsqrt_op,
    "BatchMatmulOp": bmm_op,
    "CloneOp": clone_op,
    "DivOp": div_op,
    "ExpOp": exp_op,
    "ExpandOp": expand_op,
    "VarMeanOp": var_mean_op,
    "AddMMOp": addmm_op,
    "ReshapeOp": reshape_op,
    "ViewOp": reshape_op,
    "SelectOp": select_op,
    "SliceOp": slice_op,
    "EmbeddingOp": embedding_op,
    "ConvertElementTypeOp": convert_element_type_op,
    "PermuteOp": permute_op,
    "UnsqueezeOp": unsqueeze_op,
    "TOp": t_op,
    "TransposeOp": transpose_op,
    "MaxPool2dOp": maxpool2d_op,
    "Conv2dOp": convolution2d_op,
    "ReluOp": relu_op,
    "IotaOp": iota_op,
    "SigmoidOp": sigmoid_op,
    "ReciprocalOp": reciprocal_op,
    "MeanOp": mean_op,
}

```
- 以permute_op为例
```
增加的 tosa op python frontend binding如下
def permute_op(node: PermuteOp, symbol_table):
    """
    Import the permute operation.
    From buddy graph ir's `PermuteOp` operator to MLIR TOSA `transpose`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    perm = node.args[1]
    perm_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("i", perm)))
    )
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    init_shape = ir.RankedTensorType(input_tensor.type).shape
    new_shape = []
    for perm_item in perm:
        new_shape.append(init_shape[perm_item])

    permute_result_type = ir.RankedTensorType.get(
        new_shape, result_element_type
    )
    // 根据perm调整尺寸
    // permute_result_type output
    // input_tensor input
    // perm 目标尺寸规格
    permute_op = tosa.TransposeOp(
        permute_result_type, input_tensor, perm_const_op.results[0]
    )
    return permute_op

这个permute_op操作，主要是根据node中参数获取perm，调整 symbol_table input的shape形状
```
- 

## TorchDynamoCompiler 核心算法- 的讲解
