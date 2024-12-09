# mlir 实现高性能GPU矩阵乘法
- 目标就是实现[MLIR 编译器基础设施生成高效的 GPU 代码](https://arxiv.org/pdf/2108.13191)、[实现方式](https://arxiv.org/pdf/2003.00532)

## 涉及的知识点，对于初学者的我来说，主要涉及的知识点包括
- c++ 环境以及cmake build 构建体系
- llvm/mlir构建新项目的目录组织结构
- mlir现已存在dialect体系
- 加入自定义dialect、自定义pass pipeline、自定义rewrite pattern的原则、代码组织结构
- 矩阵乘法优化的通用优化策略，比如多面体变形、循环展开
- 在gpu硬件结构层，矩阵乘法涉及的优化策略，比如共享缓存、寄存器缓存、调度流水线

## 论文解读
- 论文提出，现在市面上在研究GEMM相关优化的工作，主要有这么几个方向
```
1、开发手工私库支撑比如julia语言API库；
2、基于多面体代码生成；
3、基于triton编译器；
4、基于mlir IR基础设施，
本论文是第四种，基于mlir IR基础设施来做高性能代码生成的
```
- 相关工作源码层被提交合并到[LLVM/MLIR WMMA](https://github.com/llvm/llvm-project/commits/main/mlir/lib/Conversion/GPUToNVVM/WmmaOpsToNvvm.cpp) 相关提案中了
- llvm将nvptx的wmma api公开为instrinsics，使的mlir对tensor cores的编程成为可能，这些instrinsics和wmma api一一对应
### mlir相关的几个dialect
- **affine dialect**-多面体编译技术，使依赖分析、循环转换高校可靠
- **GPU dialect**-类似于CUDA/OpenCL的通用GPU编程范式，提供与供应商无关的抽象来模拟GPU特定的操作和属性
- **nvvm dialect**-提供了直接映射到llvm nvptx后端的操作；**llvm dialect**-mlir中最低级别的抽象
### GPU相关技术细节 
- **memory角度**看分为四层-global、l2-cache、l1-cache/shared memory、register寄存器
- **计算角度**SM与SM cores，其中SM cores包括cuda cores和tensor cores
- **编程模型角度**SM的wrap调度器、软件编程模型wrap（32个线程以锁步方式调度执行）
### 程序线程块执行的原理
- 线程块-(指定分派)->SM
- 线程块-(分解)->线程束 /wrap/32线程一束
- SM 2个Wrap 调度器-(选择分解后的线程束)->执行
- 总结来说
```
1、同一个线程块/SM 可以共享shared memory，不同线程块要通过慢的global memory共享数据
2、sync同步源语,数据写入shared memory然后由线程块所有线程读取的情况下，需要sync 同步，在读取/写入shared memory 之前，都必须sync 同步保证正确
```
### affine dialect可以完成的工作包括
- 2级循环tiling
- 创建shared memory
- padding shared memory
- 创建wmma ops
- permute/unroll展开/unroll-jam/specific loops优化
- 循环 中 全局内存加载延时隐藏优化
- 同步 barriers插入
- 向量化 load-store
- 并发探测
## 论文主要步骤
- linalg.matmul-(转换)->多面体循环/多层affine.for
- 局部/并行 tiling
- 创建/放置shared memory
- 生成wmma操作
- global memory加载延时隐藏
- 插入同步barrires
- global 到shared memory的矢量化
- 提取并行循环
- 映射到gpu 计算层次结构
- 完成延时隐藏
