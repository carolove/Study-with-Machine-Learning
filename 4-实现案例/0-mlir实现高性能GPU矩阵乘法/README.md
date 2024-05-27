# mlir 实现高性能GPU矩阵乘法
- 目标就是实现[MLIR 编译器基础设施生成高效的 GPU 代码](https://arxiv.org/pdf/2108.13191)

## 涉及的知识点，对于初学者的我来说，主要涉及的知识点包括
- [c++ 环境以及cmake build 构建体系](https://github.com/carolove/Study-with-Machine-Learning/blob/main/4-%E5%AE%9E%E7%8E%B0%E6%A1%88%E4%BE%8B/0-mlir%E5%AE%9E%E7%8E%B0%E9%AB%98%E6%80%A7%E8%83%BDGPU%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95/00-c%2B%2B%20%E7%8E%AF%E5%A2%83%E4%BB%A5%E5%8F%8Acmake%20build%20%E6%9E%84%E5%BB%BA%E4%BD%93%E7%B3%BB.md)
- [llvm/mlir构建新项目的目录组织结构](https://github.com/carolove/Study-with-Machine-Learning/blob/main/4-%E5%AE%9E%E7%8E%B0%E6%A1%88%E4%BE%8B/0-mlir%E5%AE%9E%E7%8E%B0%E9%AB%98%E6%80%A7%E8%83%BDGPU%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95/01-llvm-mlir%E6%9E%84%E5%BB%BA%E6%96%B0%E9%A1%B9%E7%9B%AE%E7%9A%84%E7%9B%AE%E5%BD%95%E7%BB%84%E7%BB%87%E7%BB%93%E6%9E%84.md)
- [mlir现已存在dialect体系]()
- [加入自定义dialect、自定义pass pipeline、自定义rewrite pattern的原则、代码组织结构等]()
- [矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等等]()
- [在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等等]()
