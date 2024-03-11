# GPU的物理结构-计算单元与内存以及缓存体系

## N卡GPU的计算单元
### N卡GPU计算单元核心名词 以4060 笔记本为例
- GPC，3个
- TPC，每个GPC有四个TPC
- SM ，每个TPC有两个SM，sm总数为24
- 共享内存，每个sm有128KB一级数据缓存/共享内存
- 处理块，每个sm有四个处理块
- CUDA core，每个sm有128个cuda core,分为四个处理块
- Warp，每个处理块有一个Warp调度器
- 寄存器文件，每个处理块有64KB寄存器文件
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
- Thread
- Block
- Grid
- Warp GPU执行程序时的调度单位，同在一个warp的线程，以不同数据资源执行相同的指令,这就是所谓 SIMT
- 半精度（FP16）
- 全精度（FP32）
