# llvm mlir toy tutorials
## 软件版本信息
- ubuntu 22.04
- gcc 11.4.0
- bazel 6.4.0
- clang 14.0.0
- python 3.10.12
- cuda 12.1
- triton release/2.2.x
- llvm-project commitid 5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372
## toy语言与AST
- ast解析后，包括这么几个元素，module-一个文件一个module，Function-对应的就是语言中的函数，Proto定义的是函数名，Block定义的是函数体
- 函数体Block中，有几个要关注的关键词，\*Op 这种是操作类，\*Decl这类是声明类，\*Expr是表达式类

### toy语言td文件中 dialect和operation
- td文件中定义了dialect以及关联operation
- 没太看明白整个cmakeflist的定义和关联关系，是如和将td构建为c++文件的？整个过程有点晕
- 接下来可以学习 https://github.com/j2kun/mlir-tutorial 这个项目用bazel来构建，是否会更清晰点，其次用bazel来重构toy 教程，也可以更清晰的了解bazel使用，以及mlir项目的结构
