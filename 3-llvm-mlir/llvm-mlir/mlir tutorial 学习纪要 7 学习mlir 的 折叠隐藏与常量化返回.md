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
- 
