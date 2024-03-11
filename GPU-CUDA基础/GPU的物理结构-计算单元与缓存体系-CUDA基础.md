# GPU的物理结构-计算单元与缓存体系-CUDA基础

## N卡GPU的计算单元
### N卡GPU计算单元核心名词 以4060 笔记本为例
- GPC，3个
- TPC，每个GPC有四个TPC
- SM ，每个TPC有两个SM，sm总数为24
- 共享内存，每个sm有128KB一级数据缓存/共享内存
- 处理块，每个sm有四个处理块
- CUDA core，每个sm有128个cuda core,分为四个处理块，整个gpu有3072个cuda core = （16 FP32 + 16 FP32/INT32 ）* 4 * 2 * 4 * 3
- Warp，每个处理块有一个Warp调度器，**一个处理块=一个warp=32个cuda core=一次cuda warp调度=32个cuda thread=一个cuda block**
- 寄存器文件，每个处理块有64KB寄存器文件，每个sm有256KB=64KB * 4
- 每个处理块，有16个专门用于FP32的CUDA Core，16个可以在FP32和INT32之间切换的CUDA Core，也就是每个处理块有32个cuda core
## N卡GPU的内存以及缓存
### CPU内存
### 全局内存
- HBM
- GPU所有SM 共享
- GPU将数据从CPU内存中load到GPU全局内存，store回CPU内存
### 共享内存 
- SRAM
- SM级缓存
- Thread Block/Streaming Multiprocessor(SM) 内多个线程间共享，用于提高线程间通信效率和数据共享速度
- 设计sm thread计算的时候，应该考虑在数据存在局部性（temporal/spatial locality）、sm block线程间数据同步协同、线程间数据共享、线程频繁访问global memory的热数据时尽量使用共享内存
- 
### 寄存器
- 线程私有级缓存

# CUDA基础
## 名词解释
- Thread，一个CUDA的并行程序会被以许多个thread来执行
- Block，数个thread会被群组成一个block，同一个block中的thread可以同步，也可以通过shared memory进行通信
- Grid，多个block则会再构成grid
- Warp GPU执行程序时的调度单位，同在一个warp的线程，以不同数据资源执行相同的指令,这就是所谓 SIMT（单指令多线程），其实是和gpu中的wrap对应，也就是sm中的计算组，每个组还有32个cuda core，也就是一个warp的单位应该为32个thread，**即由于warp的大小一般为32，所以block所含的thread的大小一般要设置为32的倍数**
- 半精度（FP16）
- 全精度（FP32）
