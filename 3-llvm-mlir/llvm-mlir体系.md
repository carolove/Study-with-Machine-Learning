# LLVM
- llvm整个体系包括编译前端-ir优化-编译后端

## 前端包括
- 预处理
- 词法分析
- 语法分析
- 语义分析=>生成AST,AST包括 声明（变量）、语句（if else while 等等）、表达（数学表达式/逻辑表达式）
- 代码生成 AST=>mlir=> LLVM-IR，到这里已经生成了module、function、baseblock、instruction
- 这一层一般都是对应Language dialect完成，比如toy语言的话，就是ToyDialect生成,在源代码中一般由对应的Language dialect的一个接口名叫做mlirGen的方法来完成CodeGen
- 本质上来说dialect就是完成这一层能够处理的operation的解释

## dialect
- a prefix namespace
- a list of types
- a list of operations,like llvm-ir's instructions
- passes, analysis\transformations（语义优化变形）\dialect conversions，pass是在编译过程中对中间表示（Intermediate Representation，IR）进行转换和优化的步骤就是一个调度manager的逻辑
- 在dialect的各个阶段包括operation中注册，pass进入pipeline流程，完成对应的优化
- 在不同的Dialect层次进行Operation转换或者做变换 = Pass
- affine这种方言为仿射变换和仿射分析提供了强大的抽象
- Vector dialect,目的是将小规模子问题进一步分解并映射到硬件寄存器和原生向量指令,即访存向量化
- Memref dialect 主要是用来做内存规划和读写。这一层的位置比较灵活， 既可以在转换成向量抽象之前，也可以在其之后
- 

### dialect的生成
- 可以借助ODS模块，使用.td文件用mlir-tblgen来自动生成dialect c++，包括dialect、operation等等
- 多找开发好的dialect包括toy dialect的td文件来阅读，提高对td文件的学习、编写能力

## IR优化
- opt可以生成IR CFG图，展示module、function、baseblock、instruction的详细图关系
- 位置在llvm/lib/transforms目录下，vectorize向量、scalar标量、
- ir的pass管理是一个独立的文件passmanagement文件中罗列，在llvm/lib/IPO/PassManagerBuilder.cpp中

## 后端
- 如果要生成本机机器码，包括寄存器分配、调度、堆栈分配
- llc本身有一部分后端lowing pass优化
- 在llvm/lib/CodeGen目录下，这里面放了一部分后端优化的pass
- llvm/lib/target还有一部分是属于各自处理器独有的pass优化，比如处理器节拍优化，多个标量是否可以合并为一个向量计算的优化等等

# MLIR
- 这个体系主要是在IR阶段

## MLIR
- 目标， 分块、融合、分配、以及向量化，对原问题进行划分并分配到不同的计算单元，之后把小规模的子问题交由 Vector dialect 来执行下一步代码生成
- MLIR 中，pattern 是最细粒度的 IR 转换工具。开发者把相关的 pattern 收集在一起并 （通过 applyPatternsAndFoldGreedily() 或者其他类似 API）在 IR 上运行来完成一个步骤
- MLIR pass 是 IR 一致性和合法性的边界
- 一个 MLIR pass 里面可以含有很多不同的步骤；内部的步骤甚至可以产生不一致的 IR。 但在 pass 运行完成后整个 IR 必须是合法的
- 比 pass 粒度更大的是 pass pipeline。Pass pipeline 组合不同的 pass 一步步完成整个代码生成流程。
