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
- 
