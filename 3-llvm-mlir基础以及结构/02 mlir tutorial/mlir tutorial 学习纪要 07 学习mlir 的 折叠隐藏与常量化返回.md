# mlir tutorial 学习纪要 7 学习mlir 的 折叠隐藏与常量化返回
- sccp 稀疏条件常量返回
- 

## sccp 稀疏条件常量返回
- 演示如下
```
原始input
func.func @test_arith_sccp() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.addi %0, %0 : i32
  %3 = arith.muli %0, %0 : i32
  %4 = arith.addi %2, %3 : i32
  return %2 : i32
}
经过learning-opt --sccp后，也就是稀疏条件 常量返回后，
func.func @test_arith_sccp() -> i32 {
  %c63_i32 = arith.constant 63 : i32
  %c49_i32 = arith.constant 49 : i32
  %c14_i32 = arith.constant 14 : i32
  %c8_i32 = arith.constant 8 : i32
  %c7_i32 = arith.constant 7 : i32
  return %c14_i32 : i32
}
再经过--canonicalize，其实也就是折叠隐藏/无用代码消除后，
func.func @test_arith_sccp() -> i32 {
  %c14_i32 = arith.constant 14 : i32
  return %c14_i32 : i32
}

```
## 多项式展开
- 要做到常量展开，需要用到一些多项式展开的操作，就是说把Mlir IR中的一部分常量计算，通过展开的方式，实现计算简化优化

## issues
- 遇到一个问题，在ops中对于add sub两个op，发现fold实现报错，应该是最新的代码依赖有问题，主要错误提示为，error: static assertion failed: PoisonAttr is undefined, either add a dependency on UB dialect or pass void as
```
通过修改依赖关系，解决了这个问题，
BUILD中引入， "@llvm-project//mlir:UBDialect",
其次，在对应的cpp代码中引入，#include "mlir/Dialect/UB/IR/UBOps.h"
这样就可以加入对于依赖，完成构建
```
- 
