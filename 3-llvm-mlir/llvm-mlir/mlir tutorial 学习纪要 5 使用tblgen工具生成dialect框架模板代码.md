# mlir tutorial 学习纪要 5 使用tblgen工具生成dialect框架模板代码
- 着手开发dialect模板，发现这dialect的td相关文件描述，不是很好懂。需要额外的文档来学，比如
```
1、Poly_Type的定义
2、typeMnemonic是什么类型
3、let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";怎么理解
4、Poly_BinOp中的，let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)"; 
```
- mlir ir输出基础格式理解
```
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("../../mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)
其中，
第一、%0  是MLIR表达式的返回值，其也是以一种和LLVM IR一样递增的临时寄存器的形式存在，但是和LLVM IR不一样的是，LLVM的Basic Block本身会占用一个临时寄存器号，BasicBlock内部的临时寄存器是从%1开始编号的，不同于MLIR。
第二、toy.transpose其中的toy类似于命名空间，算是Dialect的名字，transpose是OP的名字
第三、%arg0 : tensor<*xf64>是参数列表，前者表示参数的名字，后者表示参数的类型。
第四、to tensor<*xf64>表示的是输出参数的类型。
第五、loc("../../mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)是在添加了-mlir-print-debuginfo之后生成的调试信息，表示该行表达式是从源代码的何处产生的。
```
- assemblyFormat的理解
```
assemblyFormat，其实就是输出打印格式模板，下面的这个模板的一个输出样例就是，toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> 
let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
第一、``包裹起来的都是输出格式不会变动的内容
```
- let parameters = (ins "int":$degreeBound);的理解
```
第一、这个其实就是定义了入参，ins代表着inputs，并且，outs代表这outputs，"int"代表着 integer类型，加了$的比如$degreeBound，代表着变量
第二、类似的parser和printer，和assemblyFormat比较类似，分别定义了源码解析、输出IR打印，以及输出格式
第三、builders是指明了构造器，使用类似命令，./bazel-bin/external/llvm-project/mlir/mlir-tblgen  -gen-op-defs  lib/Dialect/Poly/PolyOps.td -I  $HOME/.cache/bazel/_bazel_username/3ded02ab66b04db8f75d57ddcaa008b1/external/llvm-raw/mlir/include -I lib/Dialect/Poly/ > ops.h，可以看到cPP的构造函数builder的生成结果
```
- 
## issues
- 我llvm-project项目使用5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372替代原始的cd5fcea6d4c70a7328ca9538c9098d9f5af69682，发现mlir的构建就出现了问题，错误码如下
```
问题：lib/Dialect/Poly/PolyOps.td:8:37: error: Couldn't find class 'Op' 需要研究为什么会这样
解答：在PolyOps.td，要加入include "mlir/IR/OpBase.td"，这样就可以了，参考该5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372的mlir examples toy中的设置就行
```
