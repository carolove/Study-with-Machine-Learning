# Matmul优化过程的学习与理解
- 主要参考文档为 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/tree/master?tab=readme-ov-file)
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

### SIMT的线程结构
### SIMT的内存缓存结构
- 全局内存
- 共享缓存
- 寄存器
### SIMT提高运算强度的优化分析
- SIMT运算强度上不来的主要原因，1）计算访存比低：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；2）访存量大：访问全局内存，C矩阵每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 2*K*M*N；
- 计算/内存/缓存设施的cycle，全局内存访问延迟高（几百cycle）、共享缓存访问延迟中（几十cycle）、寄存器访问延迟低（几个cycle），计算累加器一般为1～3个cycle完成
### SIMT提高运算强度的优化方法
- 共享缓存-避免相同位置元素被重复读取，降低访存量，直接的效果就是降低了访问延时
- 一维thread tile优化，1）降低全局内存访存量，访存量降至1/64；2） 计算访存比：引入thread tile，利用单个线程负责多个元素计算，增加计算访存比；当TM=8时，每执行共享内存As的8个次访存指令和共享内存Bs的1个访存指令，可执行8次计算指令，相比初始版本的计算访存比1:2，提高至8:9，有效隐藏访存延迟；
- 利用二维thread tile优化，1）全局访存量降低为1/128； 2）访存比也得到了相应的下降，运算强度提高1倍
- 寄存器缓存共享内存
- 向量内存指令FLOAT4优化，结合寄存器缓存优化
- 数据预取，双缓存通道，降低同步次数，将读和写分开，计算数据读取一块存储空间同时，可以同时向另一块内存写入下一轮依赖的数据，因此，只需要保证计算前待读取共享内存完成写入，即一次同步即可
