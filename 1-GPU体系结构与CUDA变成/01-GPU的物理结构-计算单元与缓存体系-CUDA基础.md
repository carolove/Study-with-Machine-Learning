# GPU的物理结构-计算单元与缓存体系-CUDA基础

## N卡GPU的计算单元
### N卡GPU计算单元核心名词 以4060 笔记本为例
- SM :sm总数为24
- 共享内存，每个sm有128KB共享内存
- Warp，每个sm有4个Warp调度器，**一个处理块=一个warp=32个cuda core=一次cuda warp调度=32个cuda thread=一个cuda block**， 24 sm * 4 wrap = 96个wrap
- sm 24 * 4 tensor core = 96个tensor core 
- 寄存器文件，每个wrap有64KB寄存器文件，每个sm有256KB=64KB * 4
- 每个wrap 处理块 有32个cuda core
## N卡GPU的内存以及缓存
### CPU内存
### 全局内存
- HBM
- GPU所有SM 共享
- GPU将数据从CPU内存中load到GPU全局内存，store回CPU内存
### 共享内存
- SM级缓存, 4 个wrap共享
### 寄存器
- wrap级，线程束私有级缓存

# CUDA基础
## 名词解释
- Thread，一个CUDA的并行程序会被以许多个thread来执行
- Block，数个thread会被群组成一个block，Block数最好以32的倍数，分解为多组wrap，同一个 block被调度在相同的sm上， 同一个block中的thread可以同步，也可以通过shared memory进行通信
- Grid，多个block则会再构成grid
- Warp GPU执行程序时的调度单位，同在一个warp的线程，以不同数据资源执行相同的指令,这就是所谓 SIMT（单指令多线程），其实是和gpu中的wrap对应，也就是sm中的计算组，每个组还有32个cuda core，也就是一个warp的单位应该为32个thread，**即由于warp的大小一般为32，所以block所含的thread的大小一般要设置为32的倍数**
- 半精度（FP16）
- 全精度（FP32）
