# mlir 实现高性能GPU矩阵乘法
- 目标就是实现[MLIR 编译器基础设施生成高效的 GPU 代码](https://arxiv.org/pdf/2108.13191)

## 涉及的知识点，对于初学者的我来说，主要涉及的知识点包括
- [c++ 环境以及cmake build 构建体系](00-c++ 环境以及cmake build 构建体系.md)
- llvm/mlir构建新项目的目录组织结构
- mlir现已存在dialect体系
- 加入自定义dialect、自定义pass pipeline、自定义rewrite pattern的原则、代码组织结构等
- 矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等等
- 在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等等
