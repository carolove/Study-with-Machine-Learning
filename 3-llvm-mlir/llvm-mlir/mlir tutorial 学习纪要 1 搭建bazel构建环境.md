# mlir tutorial 学习纪要-搭建bazel构建环境

## bazel是什么
- bazel是谷歌推出的软件构建系统，等同于cmake、makefile等
- bazel是一层python的语法糖包裹

# bazel的基本概念以及目录结构
- WORKSPACE file：WORKSPACE file就是将目录及其内容标识为Bazel工作区的文件，需要位于项目目录结构的根目录下，该文件可以为空，但是通常包含从网络或者本地文件系统中提取其他依赖项的外部代码库声明
- BUILD file：一个项目/一个目录 中包含一个或多个BUILD file，BUILD主要用于告知Bazel如何构建项目，工作区包含一个BUILD文件的目录就是一个软件包
- Bazel 的内置 cc_binary rule以及cc_library
- bazel项目中，一般bazel目录为bazel相关函数存放入口

# 语法解释
- workspace(name = "mlir_tutorial"),其中workspace，在其他软件引用该项目的时候，如果不特别指定别名的情况，@mlir_tutorial即可以外部应用访问该项目
- load("@bazel_tools 、load("//bazel:import_llvm.bzl"、等中的@、//代表着外部应用路径以及本地目录下相对路径
- bazel build @llvm-project//mlir:mlir-opt，这句话的含义是，对外部代码库（已经在加载到本地bazel cache中）llvm-project项目的mlir目录，build mlir-opt工件或者是目标，cc_binary或者cc_library目标
