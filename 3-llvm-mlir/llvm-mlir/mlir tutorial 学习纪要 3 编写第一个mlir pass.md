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
- 此次pass开发的目标：此次pass 仅仅是对这个 MLIR API 的浅包装，以展示 MLIR 基础设施，并帮助我们理解 MLIR 中 pass 的概念
- 为了实现上述目标，最通用的实现是直接通过 C++ API 实现，同时也可以用pattern rewrite engine、the dialect conversion framework、or tablegen 技术来实现同一个目标
- pattern rewrite engine、the dialect conversion framework、or tablegen等技术在后继文章中展示，展示如何用这些技术和框架来构建一个可用的全新pass
## 整个系统的个模块代码的讲解
- 有一个在本地bazelbuild有一个需要注意的事，在我本地build的时候，第一、要使用bazelisk，在本地使用.bazelversion，第二、bazel本地构建出现一些特别异常的错误，比如gcc ar什么连接错误之类的，有可能是bazel本地cache出了问题，删除bazel cache，纯净构建
- 本章节写的第一个pass：实现一个循环展开
- 
