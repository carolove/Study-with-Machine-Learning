# mlir tutorial 学习纪要-搭建bazel构建环境

## bazel是什么
- bazel是谷歌推出的软件构建系统，等同于cmake、makefile等
- bazel是一层python的语法糖包裹

# bazel的基本概念以及目录结构
- WORKSPACE file：WORKSPACE file就是将目录及其内容标识为Bazel工作区的文件，需要位于项目目录结构的根目录下，该文件可以为空，但是通常包含从网络或者本地文件系统中提取其他依赖项的外部代码库声明
- BUILD file：一个项目/一个目录 中包含一个或多个BUILD file，BUILD主要用于告知Bazel如何构建项目，工作区包含一个BUILD文件的目录就是一个软件包
- Bazel 的内置 cc_binary rule以及cc_library
- bazel项目中，一般bazel目录为bazel相关函数存放入口
