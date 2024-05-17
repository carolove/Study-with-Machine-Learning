# 理解mlir中的linalg generic 以及其中的affine map
- linalg 中的named op都是由generic 组成的，是generic的一个语法糖
- linalg 中涉及多面体计算，多面体计算，说的直白点就是多重for循环算术，参见[intro to linalg](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)

## 通用的linalg generic op的格式
```
// affine_map 中第一个(m)代表着这是一个一维的多面体，也就是只存在一层for循环，假设为二维则应该定义为(m,n)，第二个(m)代表着目标张量(ins、outs)的所以索引取值，在循环展开后代表着类似于ins[m],假设为二维张量取值(m,n)，则为ins[m][n]
#map_1d_identity = affine_map<(m) -> (m)> 

func.func @foo( %lhs : tensor<10xf32>, %rhs : tensor<10xf32>) -> tensor<10xf32> {

  %c0f32 = arith.constant 0.0 : f32
  %result_empty =  tensor.empty() : tensor<10xf32>

  // iterator_types 代表着迭代器模式，parallel代表着可并行化，reducation代表着降维
  %result = linalg.generic {
    indexing_maps=[ #map_1d_identity, #map_1d_identity,#map_1d_identity],iterator_types=["parallel"]
  } ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%result_empty : tensor<10xf32>)
  {
  // ^bb0 代表着这是一个block其实就是一个lambda 代码块，作用于迭代器上
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  } -> tensor<10xf32>
  return %result : tensor<10xf32>
}
```
```
func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
```
