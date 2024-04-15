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
- 
- 本章节写的第一个pass：实现一个循环展开
- 
## issues
- bazel cache的问题，删掉baze cache重新 纯净构建就好了
- llvm project 依赖的commit版本信息，这个mlir tutorial 的add passed这个章节依赖的commit为cd5fcea6d4c70a7328ca9538c9098d9f5af69682，但是我希望这这次学习的mlir 是为了给triton学习打下基础，我希望学习的triton 是release 2.2.x，以来的llvm commitid 为5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372，发现我如果用 5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372来构建这个章节居然不通过
```
# 具体的原因使用 在5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372这个commit中rewriter的replaceOp 出现了了问题，ambiguous，
# 后来看源代码发现原来 mlir/include/mlir/IR/PatternMatch.h这个RewriterBase的定义中有两个 virtual void replaceOp( 函数定义，其参数类型不一致，在使用这个函数调用的时候，需要将参数做显性参数传入，我是强转为了(mlir::ValueRange){newAdd}，但是我强转为(mlir::Operation *)似乎是一样的。。。
# 查看源代码，这两个声明的具体实现，我看其实干的内容是一样的，不知道为什么要加入这么一个冗余的同名函数
```
- 
