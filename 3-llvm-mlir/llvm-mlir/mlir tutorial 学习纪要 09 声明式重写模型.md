# mlir tutorial 学习纪要 9 声明式重写模型

### 重写模型 实现手段
- 通过ODS声明，然后用c++实现
- 完全用ODS来声明以及实现

#### 通过ODS声明，c++实现的方式实现 重写模型
#### 完全用ODS来声明以及实现 重写模型
- 着重研究这个逻辑
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
```
- 
