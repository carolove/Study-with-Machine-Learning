# Matmul优化过程的学习与理解
- 主要参考文档为 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
## 学习的目标
- 学习理解单指令多线程编程范式SIMT
- 学习理解SIMT范式中的线程组织结构
- 学习理解SIMT范式中的内存组织结构
- 学习理解SIMT范式中提高运算强度的优化逻辑和过程
