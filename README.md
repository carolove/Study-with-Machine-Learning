# 机器编译学习之路
- 这个项目的学习目标主要就是为了看懂并实践两篇论文
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [high performance gpu code generation for matrix-matrix multiplication using mlir](https://arxiv.org/pdf/2108.13191)

# 顶层到地层的技术依赖

## How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance技术依赖
- [c/c++ 基础以及构建基础设施cmake](https://github.com/carolove/Study-with-Machine-Learning/tree/main/1-c%2B%2B%E5%9F%BA%E7%A1%80%E4%BB%A5%E5%8F%8Acmake%E6%9E%84%E5%BB%BA%E7%8E%AF%E5%A2%83)
- [cuda程序相关依赖、编程范式以及测试](https://github.com/carolove/Study-with-Machine-Learning/tree/main/2-cuda%E7%BC%96%E7%A8%8B%E8%8C%83%E5%BC%8F-%E7%8E%AF%E5%A2%83%E4%BE%9D%E8%B5%96-%E6%B5%8B%E8%AF%95)
- [矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等](https://github.com/carolove/Study-with-Machine-Learning/tree/main/6-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E9%80%9A%E7%94%A8%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)
- [在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等](https://github.com/carolove/Study-with-Machine-Learning/tree/main/7-gpu%E7%A1%AC%E4%BB%B6%E7%BB%93%E6%9E%84%E4%BB%A5%E5%8F%8A%E7%9B%B8%E5%85%B3%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)

## high performance gpu code generation for matrix-matrix multiplication using mlir技术依赖
- [c++ 环境以及cmake build 构建体系](https://github.com/carolove/Study-with-Machine-Learning/tree/main/1-c%2B%2B%E5%9F%BA%E7%A1%80%E4%BB%A5%E5%8F%8Acmake%E6%9E%84%E5%BB%BA%E7%8E%AF%E5%A2%83)
- llvm/mlir构建新项目的目录组织结构
- [mlir现已存在dialect体系](https://github.com/carolove/Study-with-Machine-Learning/blob/main/3-llvm-mlir%E5%9F%BA%E7%A1%80%E4%BB%A5%E5%8F%8A%E7%BB%93%E6%9E%84/01%20mlir%20%E5%9F%BA%E7%A1%80%E5%AD%A6%E4%B9%A0/06%20mlir%E7%8E%B0%E5%B7%B2%E5%AD%98%E5%9C%A8dialect%E4%BD%93%E7%B3%BB.md)
- 加入自定义dialect、自定义pass pipeline、自定义rewrite pattern的原则、代码组织结构
- [矩阵乘法优化的通用优化策略，比如多面体变形、循环展开等](https://github.com/carolove/Study-with-Machine-Learning/tree/main/6-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E9%80%9A%E7%94%A8%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)
- [在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线等](https://github.com/carolove/Study-with-Machine-Learning/tree/main/7-gpu%E7%A1%AC%E4%BB%B6%E7%BB%93%E6%9E%84%E4%BB%A5%E5%8F%8A%E7%9B%B8%E5%85%B3%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)

### 额外的中文关联类似文档
- 聚焦两篇核心文档
```
https://www.lei.chat/zh/posts/mlir-linalg-dialect-and-patterns/
mlir体系的，可以从mlir角度讲述一个矩阵乘法 matmul 如何一步一步用mlir 代码 做循环展开、分块、融合、分配，并最终实现到硬件层次
https://juejin.cn/post/7008002811279441927
这篇将CUDA 矩阵乘法终极优化指南，讲述从c/c++ cuda生态圈，讲述如何将 矩阵乘法matmul 代码一步一步用 c/++c cuda  代码 做 循环展开、分块、融合、分配，并最终实现到硬件层次
```
