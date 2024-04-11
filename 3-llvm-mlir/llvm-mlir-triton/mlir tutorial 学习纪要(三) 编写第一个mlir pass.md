# mlir tutorial 学习纪要(三) 编写第一个mlir pass

## mlir pass相关概念
- dialect
- operation
- pass
- lowering

## pass在整个mlir的概念、角色
- pass的概念：pass在mlir中是一个流水线机制，pass流水线机制，串联各种dialect系统，完成dialect之间的conversion、transform以及lowering流程
- pass的角色：在mlir中pass机制，类比与工厂，mlir整个系统，相当于一个生产车间，pass机制就是整个生产车间的流水线系统，各个dialect以及各dialect定义的operation相当于流水线中的一个个环节

## mlir 系统组织目录
## mlir pass机制组织整个系统
## 整个系统的个模块代码的讲解
