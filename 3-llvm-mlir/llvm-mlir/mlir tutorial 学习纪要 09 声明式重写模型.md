# mlir tutorial 学习纪要 9 声明式重写模型

### 重写模型 实现手段
- 通过ODS声明，然后用c++实现
- 完全用ODS来声明以及实现

#### 通过ODS声明，c++实现的方式实现 重写模型
#### 完全用ODS来声明以及实现 重写模型
- 着重研究这个逻辑
```
1、先定义pattern匹配核心ODS
2、在对应的ODS Op上开启  let hasCanonicalizer = 1; 意味着可以附加 match and rewriter
3、在对于的ODS Op上的实现上加上getCanonicalizationPatterns 实现， results.add<DifferenceOfSquares>(context);意味着可以在ir结果上做 重写
```
- 代码结构
```
// 在重写匹配模型td文件中
// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)), // 基本上代表着原ir文本匹配结构
  // 代表着重写替代的ir结果
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  // 这是一个前置校验，验证是否只被使用了一次。未来其他表达式中被重复使用
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```
- 
## ISSUES
- 定义的def LiftConjThroughEval : Pat<(Poly_EvalOp $f, (ConjOp $z)),(ConjOp (Poly_EvalOp $f, $z))>，在构建的时候提示Complex.conj 匹配一个参数但是定义了两个参数，不知道如何解决
```
这个问题最后，在后继的更新中解决了
主要是在
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z)),
  (ConjOp (Poly_EvalOp $f, $z))
>;
改造为
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
应该是这个conj需要加入第二个入参
```
## 读懂mlir语法
- eg
```
complex.conj (complex::ConjOp)
1、Syntax:
operation ::= `complex.conj` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
2、Example:
%a = complex.conj %b: complex<f32>
3、
|Attribute	|    MLIR Type	                   |   Description |
|fastmath	    |  ::mlir::arith::FastMathFlagsAttr| 	Floating point fast math flags|
```
- 
