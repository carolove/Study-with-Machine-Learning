# mlir tutorial 加入toy 构建
- 将toy项目加入到mlir tutorial项目中，学习mlir，学习bazel，学习使用将toy装到这个项目中

## 第一篇，bazel构建
- 将llvm-project 的commit修改修改到5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372，因为我想使用triton的release 2.2.x，我要将llvm修改为这个commit，build： bazel run @llvm-project//mlir:mlir-opt -- --help
- 第一个例子是mlir ast级，bazel-bin/external/llvm-project/mlir/mlir-opt ctlz.mlir -- $(pwd)/ctlz.mlir ，这是最原始的mlir 代码
- 第二个例子是lowering级，bazel-bin/external/llvm-project/mlir/mlir-opt --convert-math-to-funcs=convert-ctlz $(pwd)/ctlz.mlir，这个是进行了第一次lowering的代码
- 第一堂课中的bazel相关配置，包括WORKSPACE、bazel目录下的函数，都仅仅是加载了llvmproject到bazel cache中，真正执行bazel build，比如bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/ctlz.mlir，其实是在build cache系统缓存代码库中的llvm project项目的build 构建

# 第二篇，bazel测试
- 主要讲述在llvm mlir的测试框架，使用lit以及filecheck等语法，综合bazel的测试框架，来完成整个测试过程，将测试mlir语法和lowering结构写在一个测试文件中，调用lit filecheck来验证 完成单元测试
