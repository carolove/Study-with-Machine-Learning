# mlir tutorial 学习纪要(二) 搭建llvm mlir测试环境

## 测试环境相关概念
- Lit,这个工具会加载项目中的lit.py相关配置文件，做好路径准备，并读取测试文件，执行RUN:注释部分内容，完成自动测试
- FileCheck：会对测试结果做比对，// CHECK: 以及// CHECK-NOT: 作为起止符
- bazel test //...，会找到tests目录下的BUILD文件，  glob_lit_tests 宏, 以及 filegroup函数定义了所有要参与测试的工具集以及测试文件定义，最后执行整个测试
