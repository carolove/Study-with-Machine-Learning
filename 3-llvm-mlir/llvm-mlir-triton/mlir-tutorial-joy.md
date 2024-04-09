# mlir tutorial 加入toy 构建
- 将toy项目加入到mlir tutorial项目中，学习mlir，学习bazel，学习使用将toy装到这个项目中

## bazel构建
- 将llvm-project 的commit修改修改到5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372，因为我想使用triton的release 2.2.x，我要将llvm修改为这个commit，build： bazel run @llvm-project//mlir:mlir-opt -- --help
- 第一个例子是mlir ast级，bazel-bin/external/llvm-project/mlir/mlir-opt ctlz.mlir -- $(pwd)/ctlz.mlir ，这是最原始的mlir 代码
- 第二个例子是lowering级，bazel-bin/external/llvm-project/mlir/mlir-opt --convert-math-to-funcs=convert-ctlz $(pwd)/ctlz.mlir，这个是进行了第一次lowering的代码
- 
