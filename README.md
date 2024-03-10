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
- [构建编译](https://github.com/carolove/Study-with-Machine-Learning/blob/main/llvm-mlir-triton/triton%E5%AD%A6%E4%B9%A0%EF%BC%88%E4%B8%80%EF%BC%89llvm%5Ctriton%E6%9E%84%E5%BB%BA%E7%BC%96%E8%AF%91.md)
- 学习triton 的python DSL
- 学习triton的jit 相关
- 用triton jit学习实现基础算子（可能要拆的更细才行）
- 源码解释triton运行、llvm ir下降过程、机器码生成过程
