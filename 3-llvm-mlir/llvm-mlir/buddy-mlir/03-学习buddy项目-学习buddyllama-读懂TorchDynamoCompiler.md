# 读懂TorchDynamoCompiler
- 目标 读懂TorchDynamoCompiler，了解从llm 模型导入到mlir ir生成的逻辑，知其然知其所以然

## TorchDynamoCompiler 核心算法以及数据结构
- 核心commit [TorchDynamo compiler ](https://github.com/buddy-compiler/buddy-mlir/pull/208)
- 核心的函数为from buddy.compiler.frontend import DynamoCompiler ，DynamoCompiler的 importer可以将model graph 转换为 mlir ir，算子替换主要由 primary_registry=tosa.ops_registry
- 需要将pytorch 的[aten ir](https://pytorch.org/docs/master/ir.html#core-aten-ir) convert到咱们自定义的primary registry ops上
- 整个Tensor Operator Set Architecture [tosa](https://mlir.llvm.org/docs/Dialects/TOSA/) op在mlir dialect定义的所有op
- 整个mlir 体系关于大模型机器编译过程涉及的dialect
```
mlir 体系关于大模型机器编译过程涉及的dialect，如下图,其中核心dialect，
在最顶层，tf、tflite、以及 torch 等 dialect 用于机器学习框架的接入； mhlo 和 tosa dialect 则将来自各种框架的五花八门的算子集 (op set) 收缩整合， 转化成统一的表示，作为下层 MLIR 代码生成栈的输入程序。
在其下一层，linalg dialect 主要用来对原问题分块 (tiling) 并映射到硬件计算体系 (compute hierarchy)。
Memref dialect 这一层主要是用来做内存规划和读写。这一层的位置比较灵活， 既可以在转换成向量抽象之前，也可以在其之后。
最底层有 llvm 以及 spirv dialect，转换到这一层是为调用 LLVM 编译器栈做进一步的更底层的代码生成，或者产生最终的程序 SPIR-V 二进制表示。
```

![image](https://github.com/carolove/Study-with-Machine-Learning/assets/834467/1207e7fe-ec29-4acf-8fb9-47fc63320ac9)


## TorchDynamoCompiler 核心算法- 的讲解
- 这个提交主要的作用就是利用TorchDynamo, a Python-level JIT compiler introduced in PyTorch 2.0. Using this importer, one can convert a PyTorch function/model to corresponding MLIR code.
- 需要实现的op的原有，What this importer do is to convert a piece of PyTorch code to the corresponding MLIR code. To achieve it, we write some conversion functions that map PyTorch's operators to MLIR code snippets. Currently, we've mapped about 20 operators. For what operators are supported, please refer to the [frontend/Python/ops](https://github.com/buddy-compiler/buddy-mlir/tree/main/frontend/Python/ops) directory.
- primary registry的原有， When importer is going to import a PyTorch operator, it will first search the primary registry for the operator's mapping function. If the operator is not found in the primary registry, the importer will try to search the fallback registry. By default, the importer will use `tosa` registry as the primary registry, and all the other registries as the fallback registry.
- 如何通过torch aten ir来构建模型的[fx graph](https://pytorch.org/docs/stable/torch.compiler_transformations.html)
- 整个系统流程为 TorchDynamo + llm model -> fx graph = aten ir level -> + primary registry ops -> mlir ir 
## 材料
- [buddy compiler llama 端到端的编译](https://zhuanlan.zhihu.com/p/665429695) 这篇文章已经可以完全理解
