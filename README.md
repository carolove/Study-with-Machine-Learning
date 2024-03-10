# 机器学习之路
## GPU/CUDA基础
- GPU的结构-sm/tensor core/全局内存/共享内存/寄存器
- CUDA基础-thread/全局内存/共享内存/寄存器/合并内存/bank conflict
## 基础算子
- add
- softmax
- matmul
- attention
## CUDA实现基础算子
- cuda环境、构建、运行、验证，实现add算子
- cuda实现softmax算子
- cuda实现matmul算子
- 学习cuda attention算法
## CUDA优化SGEMM
## LLVM/MLIR/triton
- 构建编译
- 学习triton 的python DSL
- 学习triton的jit 相关
- 用triton jit学习实现基础算子（可能要拆的更细才行）
- 源码解释triton运行、llvm ir下降过程、机器码生成过程
