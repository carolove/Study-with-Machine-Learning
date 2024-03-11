# GPU的物理结构-计算单元与内存以及缓存体系

## N卡GPU的计算单元
### N卡GPU计算单元核心名词
- Thread
- SM
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
