# Matmul优化过程的学习与理解
- 主要参考文档为 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
## 学习的目标
- 学习理解单指令多线程编程范式SIMT
- 学习理解SIMT范式中的线程组织结构
- 学习理解SIMT范式中的内存组织结构
- 学习理解SIMT范式中提高运算强度的优化逻辑和过程
### 学习理解SIMT范式
- SIMT范式在编码上看起来是一个单线程模式的代码
```c++
  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
```
- SIMT在编译器中被优化为了多线程执行模式，上述的单线程代码块类似于一个template 模板，在模板代码执行前会计算实际运行线程的索引（x,y），以此多线程执行时各自加载对应索引的数据来互不干涉的运行
```c++
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
  // 单线程模板代码
  }

```
- SIMT中的索引运算用到的blockIdx、blockDim、threadIdx通过全局获取以及范式传入
