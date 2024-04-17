# mlir tutorial 学习纪要 4 使用tblgen工具生成pass框架模板代码
- 主要的参考文章为[MLIR — Using Tablegen for Passes](https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)
- 前三章请参考本系列文章，以及[icebear的知乎专栏](https://www.zhihu.com/column/c_1711859337423855616)，里面前三章的翻译说明
- 本文也不是逐文翻译，谨作为参考

## 第一节
- 主要讲述前三章的学习中，我们大致上已经了解到了mlir的运行机制，以及mlir pass框架下的主要核心api，通过直接实现mlir pass框架api的方式，是可以完成一个pass的构造的，但是在实际的工程实践中，这种方式并不是主流，主流的方式是通过使用mlir生态提供的code gen工具来完成，即通过tblgen工具来生成mlir pass/mlir dialect相关的框架模板代码，这个章节我们就用tblgen工具来完成模板代码的生成

## 第二节
- 
