# mlir tutorial 学习纪要 4 使用tblgen工具生成pass框架模板代码
- 主要的参考文章为[MLIR — Using Tablegen for Passes](https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)
- 前三章请参考本系列文章，以及[icebear的知乎专栏](https://www.zhihu.com/column/c_1711859337423855616)，里面前三章的翻译说明
- 本文也不是逐文翻译，谨作为参考

## 第一节
- 主要讲述前三章的学习中，我们大致上已经了解到了mlir的运行机制，以及mlir pass框架下的主要核心api，通过直接实现mlir pass框架api的方式，是可以完成一个pass的构造的，但是在实际的工程实践中，这种方式并不是主流，主流的方式是通过使用mlir生态提供的code gen工具来完成，即通过tblgen工具来生成mlir pass/mlir dialect相关的框架模板代码，这个章节我们就用tblgen工具来完成模板代码的生成

## 第二节 How to think about tablegen，如何正确认识tblgen工具
- 这节中作者其实对tblgen工具是有点微词的，主要体现在
```
第一、用mlir tblgen生成的框架代码的错误信息提示有点简陋和模糊，不容易在开发中定位到问题所在，这个mlir框架下的程序一旦构建失败，会产生上百行的错误日志，有点迷糊，不知所错；
第二、其次是用tblgen生成的框架模板代码，没有清晰的标注出那些接口或者那些代码块是需要重构的，对于一个新手来说，不够友好，很多时候并没有清晰的提示以及文档来标注，生成后的代码，哪些部分不需要修改，哪些部分需要重写，哪些部分需要实现，而且这方面的文档更新的太不及时了，甚至有些最新代码构建后的mlir tblgen生成的模板代码的细节说明，还需要去查阅github commit修改记录、需要去论坛找才可以找到修改原因，才能知道如何改造和填充模板代码；
```

## 第三节 如何使用tblgen工具
- 这节作者打算用tblgen工具重新生成第三章所开发的AffineFullUnroll pass
- 这个commit，[add tablegen for loop unrolling passes](https://github.com/j2kun/mlir-tutorial/pull/7/commits/d5f5a0d9cc909351076fba97b38708215cd83585)，说明了如何增加一个tblgen生成的pass，有这么几个代码细节需要说明
```

```
- 

## 第四节 附加章节 在bazel工程中使用制定python环境，并安装python 环境依赖
- 这个章节应该前置到上一个章节的，这个commit [Use a hermetic python and install lit](https://github.com/j2kun/mlir-tutorial/commit/aac84908f7b09ec1b14489bbc0837e697b191630)，主要的作用就是在bazel工程中增加制定版本的python安装，并设置该python环境的依赖
```
主要的代码细节如下
主要在WORKSPACE中，引入git相关以来下载rules_python，这样的用这个库的函数，可以轻松完成python 3.10的下载、运行以及pip环境等操作
```
