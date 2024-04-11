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
- 原始mlir的目录组织如下
```
include 存放头文件 tablegen
  Transform/  存放 dialect 内转换代码的 passes 的文件
  Conversion/  存放 dialect 间转换代码的 passes 的文件
  Analysis/ 存放用于分析 passes
lib  放置实现代码
  Transform/  存放 dialect 内转换代码的 passes 的文件
  Conversion/  存放 dialect 间转换代码的 passes 的文件
  Analysis/ 存放用于分析 passes
```
- 在bazel项目中，修改两层目录合并到一层都为lib
```
lib
│   └── Transform
│       └── Affine
│           ├── AffineFullUnroll.cpp
│           ├── AffineFullUnroll.h
│           └── BUILD
```
- 目的是原始的目录设计是为了隐藏实现lib，只暴露公共接口include，在bazel构建系统中，有足够多的工具和方法实现  这个目标，因此不需要如此复杂的设计目录结构
## mlir pass机制组织整个系统
## 整个系统的个模块代码的讲解
