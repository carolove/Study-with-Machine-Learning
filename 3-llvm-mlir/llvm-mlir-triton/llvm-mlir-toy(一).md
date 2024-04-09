# llvm mlir toy tutorials第一堂课

## toy语言与AST
- ast解析后，包括这么几个元素，module-一个文件一个module，Function-对应的就是语言中的函数，Proto定义的是函数名，Block定义的是函数体
- 函数体Block中，有几个要关注的关键词，\*Op 这种是操作类，\*Decl这类是声明类，\*Expr是表达式类
