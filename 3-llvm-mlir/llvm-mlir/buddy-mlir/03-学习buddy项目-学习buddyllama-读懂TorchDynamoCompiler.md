# 读懂TorchDynamoCompiler
- 目标 读懂TorchDynamoCompiler，了解从llm 模型导入到mlir ir生成的逻辑，知其然知其所以然

## TorchDynamoCompiler 核心算法以及数据结构
- 核心commit [TorchDynamo compiler ](https://github.com/buddy-compiler/buddy-mlir/pull/208)
- 核心的函数为from buddy.compiler.frontend import DynamoCompiler ，DynamoCompiler的 importer可以将model graph 转换为 mlir ir，算子替换主要由 primary_registry=tosa.ops_registry
- 需要将pytorch 的[aten ir](https://pytorch.org/docs/master/ir.html#core-aten-ir) convert到咱们自定义的primary registry ops上
- 整个Tensor Operator Set Architecture [tosa](https://mlir.llvm.org/docs/Dialects/TOSA/) op在mlir dialect定义的所有op
- 整个mlir 体系关于大模型机器编译过程涉及的dialect
```
mlir 体系关于大模型机器编译过程涉及的dialect，如下图
其中核心dialect，
PyTorch 的 torch dialect
tosa dialect
linalg 完美嵌套循环 (perfect loop nest)
LLVM IR 或者 SPIR-V，通常都是完整 (complete) 的； 它们包含所需的所有指令来表示整个 CPU 后者 GPU 程序
```

![image](https://github.com/carolove/Study-with-Machine-Learning/assets/834467/1207e7fe-ec29-4acf-8fb9-47fc63320ac9)
- 这个项目需要的tosa的wrapper为
```
frontend/Python/ops/tosa.py中注册的

ops_registry = {
    "AddOp": add_op,
    "MulOp": mul_op,
    "SubOp": sub_op,
    "SumDimOp": sum_op,
    "TanhOp": tanh_op 
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
这些算子就是我们在mkir ir生成中要用到的算子，也是从高级dsl降级的目标算子

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
- 这个提交主要的作用就是利用TorchDynamo, a Python-level JIT compiler introduced in PyTorch 2.0. Using this importer, one can convert a PyTorch function/model to corresponding MLIR code.
- 需要实现的op的原有，What this importer do is to convert a piece of PyTorch code to the corresponding MLIR code. To achieve it, we write some conversion functions that map PyTorch's operators to MLIR code snippets. Currently, we've mapped about 20 operators. For what operators are supported, please refer to the [frontend/Python/ops](https://github.com/buddy-compiler/buddy-mlir/tree/main/frontend/Python/ops) directory.
- primary registry的原有， When importer is going to import a PyTorch operator, it will first search the primary registry for the operator's mapping function. If the operator is not found in the primary registry, the importer will try to search the fallback registry. By default, the importer will use `tosa` registry as the primary registry, and all the other registries as the fallback registry.
- 如何通过torch aten ir来构建模型的[fx graph](https://pytorch.org/docs/stable/torch.compiler_transformations.html)
- 整个系统流程为 TorchDynamo + llm model -> fx graph = aten ir level -> + primary registry ops -> mlir ir 
## 材料
- [buddy compiler llama 端到端的编译](https://zhuanlan.zhihu.com/p/665429695) 这篇文章已经可以完全理解
