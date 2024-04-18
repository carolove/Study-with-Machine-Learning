# mlir tutorial 学习纪要 5 使用tblgen工具生成dialect框架模板代码
- 着手开发dialect模板，发现这dialect的td相关文件描述，不是很好懂。需要额外的文档来学，比如
```
1、Poly_Type的定义
2、typeMnemonic是什么类型
3、let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";怎么理解
4、Poly_BinOp中的，let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)"; 每一个都不容易理解，有点像天书
```
- 我llvm-project项目使用5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372替代原始的cd5fcea6d4c70a7328ca9538c9098d9f5af69682，发现mlir的构建就出现了问题，错误码如下
```
lib/Dialect/Poly/PolyOps.td:8:37: error: Couldn't find class 'Op'
需要研究为什么会这样
```
