# GPU的物理结构-计算单元与内存以及缓存体系

## N卡GPU的计算单元
### N卡GPU计算单元核心名词
- Thread
- SM

## N卡GPU的内存以及缓存
- CPU内存
- 全局内存 HBM，GPU所有SM 共享，GPU将数据从CPU内存中load到GPU全局内存，store回CPU内存
- 共享内存 SRAM，GPU多处理器 BLOCK SM内线程共享，属于SM级缓存
- 寄存器 是线程私有级缓存
