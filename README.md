# 机器编译学习之路
- 这个项目的学习目标主要就是为了看懂并实践两篇论文
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [high performance gpu code generation for matrix-matrix multiplication using mlir](https://arxiv.org/pdf/2108.13191)、[HIGH PERFORMANCE CODE GENERATION IN MLIR](https://arxiv.org/pdf/2003.00532)

# 顶层到地层的技术依赖

## How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance技术依赖
- c/c++ 基础以及构建基础设施cmake
- cuda程序相关依赖、编程范式以及测试
- 矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等
- 在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等

## high performance gpu code generation for matrix-matrix multiplication using mlir技术依赖
- c++ 环境以及cmake build 构建体系
- llvm/mlir构建新项目的目录
- mlir现已存在dialect体系
- 加入自定义dialect、自定义pass pipeline、自定义rewrite pattern的原则、代码组织结构
- 矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等
- 在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等

### 额外的中文关联类似文档
- 聚焦两篇核心文档
```
https://www.lei.chat/zh/posts/mlir-linalg-dialect-and-patterns/
mlir体系的，可以从mlir角度讲述一个矩阵乘法 matmul 如何一步一步用mlir 代码 做循环展开、分块、融合、分配，并最终实现到硬件层次
https://juejin.cn/post/7008002811279441927
这篇将CUDA 矩阵乘法终极优化指南，讲述从c/c++ cuda生态圈，讲述如何将 矩阵乘法matmul 代码一步一步用 c/++c cuda  代码 做 循环展开、分块、融合、分配，并最终实现到硬件层次
```
