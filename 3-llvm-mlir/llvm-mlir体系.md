# LLVM
- llvm整个体系包括编译前端-ir优化-编译后端

## 前端包括
- 预处理
- 词法分析
- 语法分析
- 语义分析=>生成AST,AST包括 声明（变量）、语句（if else while 等等）、表达（数学表达式/逻辑表达式）
- 代码生成 AST=> LLVM-IR

## IR优化
- opt可以生成IR CFG图，展示module、function、baseblock、instruction的详细图关系
- 位置在llvm/lib/transforms目录下，vectorize向量、scalar标量、
- ir的pass管理是一个独立的文件passmanagement文件中罗列，在llvm/lib/IPO/PassManagerBuilder.cpp中

## 后端
- 如果要生成本机机器码，包括寄存器分配、调度、堆栈分配
- llc本身有一部分后端lowing pass优化
- 在llvm/lib/CodeGen目录下，这里面放了一部分后端优化的pass
- llvm/lib/target还有一部分是属于各自处理器独有的pass优化，比如处理器节拍优化，多个标量是否可以合并为一个向量计算的优化等等
