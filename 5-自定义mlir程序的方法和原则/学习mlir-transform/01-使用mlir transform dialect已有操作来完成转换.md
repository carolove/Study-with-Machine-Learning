# 使用mlir transform dialect已有操作来完成转换

## 介绍 
```
变换方言transform dialect的变换行为可以精确地定位到 方言语句/IR 中的特定操作/匹配行，并将它们串连起来。我们将这些变换操作产生的 IR 称为变换 IR。我们将正在进行变换操作的 IR 称为有效载荷 IR。

变换 IR 操作对可能与有效载荷 IR 的操作Op、值或属性相关联的值进行变换。我们将前两种分别称为操作和值句柄。我们将最后一种称为参数。

变换 IR 的应用总是从一个顶层操作开始。在 C++ API 中，此操作传递给函数applyTransforms。此顶层操作指定是否应执行其他变换以及如何执行。最常见的顶层操作transform.named_sequence只是依次应用其主体中列出的其他变换操作，类似于函数或宏。

让我们通过常见的“全连接 + 偏差 + ReLU” ML 层上的简单转换序列来说明这一点，该序列归结为执行矩阵乘法，然后进行（逐元素）矩阵加法并取元素最大值 0。这可以使用以下 IR 来表示：

func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
```
## 顶级序列操作 
```
出于性能原因，我们希望平铺Tiling和融合Fusion这些操作以利用缓存局部性。这是需要逐个执行的转换序列，因此我们自然会从相应的顶级转换操作开始。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">):
    transform.yield
  }
}
这次行动有几个方面值得关注。

它的特殊名称@__transform_main和第一个参数由解释器传递强制执行，类似于 C 程序的入口点需要被调用的方式，main并且可能具有签名。此参数将与顶级有效负载操作相关联，通常是传递所应用的操作。请注意，当通过或以编程方式int (int argc, char** argv)应用转换时，这些都不是必需的。applyTransformsapplyNamedSequence

其余的入口块参数是可选的，可以与序列中有用的有效负载属性、操作或值相关联。这些也是在调用时指定的applyTransforms。在我们的例子中，我们对要平铺Tiling和融合Fusion的矩阵乘法和元素运算感兴趣。

所有值句柄都具有 Transform 方言类型。这些类型指定与其关联的有效载荷 IR 实体的某些属性。在此示例中，transform.any_op表示句柄与任意有效载荷操作相关联。相反，transform.op<"X">表示句柄仅与类型的有效载荷操作相关联X。在创建句柄/有效载荷关联时会验证这些约束。对于顶级转换操作的入口块参数，这在函数的早期发生applyTransforms。如果不满足约束，转换应用程序将失败并为用户生成诊断信息。

最后，该操作被包装在一个模块中，transform.with_named_sequence如果存在多个命名序列，则该模块会触发所有必要的验证。
```
## 故障传播 
```
Transform 方言基础结构具有一种处理支持可恢复错误的诊断的特定机制。通过考虑具有指定故障传播模式的强制属性的（未命名）序列操作，可以最好地理解这一点。有两种选择：

如果任何嵌套转换失败，“propagate” 也会导致序列转换失败；
即使其中一个嵌套转换失败，“suppress” 也会使序列成功，但不会尝试执行序列中失败转换之后的转换。
后者允许围绕序列的转换脚本继续运行，尽管序列中存在错误，但前提是这些错误是可以恢复的。由于我们只构建转换脚本，因此最好传播故障，这样我们就知道什么时候有些事情不适用。

要检查或调试变换序列，可以打印与变换 IR 值相关的各种实体。例如，我们可以打印与句柄相关的操作：

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.debug.emit_remark_at %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.debug.emit_remark_at %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
```
## 变换方言解释器 
```
由于我们不想在每次更改转换时都重新编译编译器，因此我们可以使用 Transform 方言解释器传递将此转换序列应用于有效载荷 IR。正如我们将在下一章中看到的那样，可以定义自定义传递，甚至可以将转换解释器集成到更大的传递中。现在，我们可以使用现有的测试传递：

$ mlir-opt sequence.mlir --pass-pipeline="
    builtin.module(transform-interpreter{
        debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"
该sequence.mlir文件包含嵌套在同一模块中的有效载荷 IR 函数和变换 IR 序列。变换解释器传递将命名序列应用于传递的锚点操作。在我们的@__transform_main例子中，我们还要求解释器传递通过相应的传递选项将顶级序列的两个额外参数与所有linalg.matmul和linalg.elemwise_binary有效载荷操作相关联。运行此传递会产生预期的注释：

sequence.mlir:7:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
sequence.mlir:7:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
sequence.mlir:10:13: remark: elemwise_binaries
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
sequence.mlir:10:13: note: see current operation: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
sequence.mlir:14:13: remark: elemwise_binaries
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
sequence.mlir:14:13: note: see current operation: %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
请注意，%arg2与两个元素有效负载操作相关联。任何句柄都与实体列表相关联。各个转换可能会或可能不会关心该列表中元素的顺序。
```

## 指定转换 
```
现在我们已经掌握了要转换的操作，可以开始应用转换了。首先，让我们尝试平铺Tiling matmul 操作本身。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // The actual tiling transformation takes tile sizes as attributes.
    %loop, %tiled = transform.structured.tile_using_forall %arg1
                    tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">)
     -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
转换返回两个句柄，如其 文档中所示：

linalg.generic对原始数据子集进行操作的句柄。
scf.forall张量周围的“多重 for”循环的句柄。
使用与上面相同的命令运行此转换预计会产生平铺Tiling代码。

func.func @fc_relu(%arg0: tensor<512x512xf32>,
                   %arg1: tensor<512x512xf32>,
                   %arg2: tensor<512x512xf32>,
                   %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
    %4 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg5)
    %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1]
                     : tensor<512x512xf32> to tensor<4x512xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1]
                       : tensor<512x512xf32> to tensor<512x32xf32>
    %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1]
                      : tensor<512x512xf32> to tensor<4x32xf32>
    %5 = linalg.matmul
         ins(%extracted_slice, %extracted_slice_0
             : tensor<4x512xf32>, tensor<512x32xf32>)
         outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1]
          : tensor<4x32xf32> into tensor<512x512xf32>
    }
  }
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
    ins(%1, %cst : tensor<512x512xf32>, f32)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %2 : tensor<512x512xf32>
}
除了生成新句柄之外，平铺Tiling变换操作还会消耗操作数句柄。这意味着句柄在此操作之后将失效，并且不应再使用。变换操作需要将其所有操作数标记为已消耗或只读。如果关联的有效载荷操作被擦除或重新创建（这意味着被擦除并以类似的结构重新创建），变换操作通常会消耗操作数。由于句柄本质上是对有效载荷操作的引用，因此如果有效载荷不再存在，它们将变为悬空。
```
## 处理无效和昂贵检查模式 
```
未定义行为一旦发生，就很难处理，因此 Transform 方言解释器默认执行一组额外的、可能很昂贵的检查，以检测转换 IR 中的大多数未定义行为。例如，如果我们想在句柄%arg1被使用后使用它，这将导致未定义行为，在调试版本中表现为断言，在发布模式下则可能表现为分段错误。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // The actual tiling transformation takes tile sizes as attributes.
    %loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
        : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

    // This is trying to use an invalidated handle leading to undefined behavior.
    transform.debug.emit_remark_at %arg1, "remark" : !transform.op<"linalg.matmul">
    transform.yield
  }
}
但是，通过在解释器中启用昂贵的检查，可以产生一个很好的诊断结果：

sequence.mlir:28:3: error: op uses a handle invalidated by a previously executed transform op
  transform.debug.emit_remark_at %mm, "elemwise_binaries" : !transform.any_op
  ^
sequence.mlir:26:9: note: handle to invalidated ops
  %mm = transform.cast %matmul : !transform.op<"linalg.matmul"> to !transform.any_op
        ^
sequence.mlir:27:19: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %loop, %tiled = transform.structured.tile_using_forall %mm tile_sizes [4, 32]
当编译时性能是一个问题，并且转换序列足够稳定时，可以通过提供disable-expensive-checks传递选项或在TransformOptions传递中设置相应的标志来禁用解释器中的昂贵检查，以提高性能applyTransforms。

你可能会发现，某些操作（例如）transform.cast不会消耗操作数（因为它们不会擦除相应的操作）。那么如果我们尝试使用该操作数，会发生什么情况呢？

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // We can cast one type to another as long as operations are compatible
    // with both types. This creates "aliasing" handles.
    %casted = transform.cast %arg1 : !transform.op<"linalg.matmul">
        to !transform.any_op

    // The actual tiling transformation takes tile sizes as attributes.
    %loop, %tiled = transform.structured.tile_using_forall %arg1
                    tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">)
     -> (!transform.any_op, !transform.any_op)

    // Consuming an operand invalidates the consumed handle and any other handle
    // that is associated with the same payload operations, or payload
    // operations nested in them.
    transform.debug.emit_remark_at %casted, "remark"
      : !transform.any_op
    transform.yield
  }
}
%arg1和都%casted引用相同的有效负载操作。扩展引用类比，这些引用互为别名。自然，当有效负载操作被擦除时，对它的所有引用都会变成悬空。句柄也是如此。事实上，使用操作数会使操作数句柄以及与任何相同有效负载操作相关联的任何其他句柄无效。有效负载 IR 考虑是递归的：与嵌套在已擦除操作中的有效负载操作相关联的句柄也会失效（因为擦除操作也会擦除其区域和所有包含的操作）。昂贵的检查模式也可以处理这种情况。

sequence.mlir:28:3: error: op uses a handle invalidated by a previously executed transform op
  transform.debug.emit_remark_at %matmul, "elemwise_binaries" : !transform.op<"linalg.matmul">
  ^
sequence.mlir:21:29: note: handle to invalidated ops
^bb0(%root: !transform.any_op, %matmul: !transform.op<"linalg.matmul">, %elemwise: !transform.op<"linalg.elemwise_binary">):
                            ^
sequence.mlir:27:19: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %loop, %tiled = transform.structured.tile_using_forall %mm tile_sizes [4, 32]
```
## 使用句柄链接变换 
```
回到变换序列，我们已经平铺Tiling了矩阵乘法，但我们也希望平铺Tiling和融合Fusion元素操作。在结构化操作范式中，典型的做法是平铺Tiling某个非循环数据流图中的最后一个操作，然后逐步融合Fusion产生其操作数的操作。这样就无需明确平铺Tiling所有操作，因为融合Fusion可以调整它们的大小并在需要时注入重新计算。因此，我们不会平铺Tiling matmul 操作，而是平铺Tiling链中的最后一个操作，然后将前面的操作融合Fusion到平铺Tiling产生的循环中。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2
        : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)

    // The actual tiling transformation takes tile sizes as attributes. It
    // produces a handle to the loop generated during tiling.
    %tiled_max, %loop =
        transform.structured.tile_using_forall %max tile_sizes [8, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one by one. This requires the operation that is being fused to
    // define the value used within the loop, so the order of such fusions is
    // important. We could also use "transform.merge_handles" to obtain a single
    // handle to all operations and give it to `fuse_into_containing_op` that
    // would take care of the ordering in this case.
    %add_fused, %loop_0 =
        transform.structured.fuse_into_containing_op %add into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_1 =
        transform.structured.fuse_into_containing_op %arg1 into %loop_0
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
这样就实现了所需的平铺Tiling和融合Fusion。
```
## 更多处理失效 
```
最后，我们假设存在一个高效的微内核，或者说一个以内在函数表示的硬件指令，用于 4x4 矩阵乘法。为此，我们需要将融合Fusion操作平铺Tiling到所需大小，然后对其进行概述。然后可以将生成的函数调用替换为对微内核的调用。

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2
        : (!transform.op<"linalg.elemwise_binary">)
          -> (!transform.any_op, !transform.any_op)

    // The actual tiling transformation takes tile sizes as attributes. It
    // produces a handle to the loop generated during tiling.
    %tiled, %loop = transform.structured.tile_using_forall %max
                    tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one by one. This requires the operation that is being fused to
    // define the value used within the loop, so the order of such fusions is
    // important. We could also use "transform.merge_handles" to obtain a single
    // handle to all operations and give it to `fuse_into_containing_op` that
    // would take care of the ordering in this case.
    %add_fused, %loop_0 =
        transform.structured.fuse_into_containing_op %add into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_1 =
        transform.structured.fuse_into_containing_op %arg1 into %loop_0
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // Tile again to get the desired size. Note that this time this tiles the
    // "add" operation and fuses matmul into the loop, but doesn't affect the
    // "max" operation. This illustrates the precise targeting with the
    // transform dialect. Otherwise, it is difficult to differentiate "add" and
    // "max", both of which having the same kind.
    %tiled_2, %loop_2 =
        transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused_2, %loop_3 =
        transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // Since outlining is currently only implemented for region-holding
    // operations such as loops, use tiling to size 1 to materialize the outer
    // loop that is going to be outlined.
    %_, %outline_target =
        transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %matmul_fused_2
        into %outline_target
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %func, %call = transform.loop.outline %outline_target
                   {func_name = "outlined"}
        : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

    transform.yield
  }
}
此附加转换还说明了嵌套操作的句柄失效。操作transform.loop.outline使用循环的句柄，从而使循环及其中嵌套的任何操作（例如）的句柄失效%2。尝试使用此句柄将导致未定义的行为。（请注意，这种特定形式的大纲不一定需要使用操作数，因为实现仅移动区域而不重新创建操作，但转换的作者还是选择使句柄失效。）
尝试在概述后访问融合Fusion结果会产生以下错误

test/Examples/transform/Ch1/invalidation-2.mlir:109:3: error: op uses a handle invalidated by a previously executed transform op
  transform.debug.emit_remark_at %outline_target, "outlined loop" : !transform.any_op
  ^
test/Examples/transform/Ch1/invalidation-2.mlir:102:25: note: handle to invalidated ops
  %outline_target, %_ = transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
                        ^
test/Examples/transform/Ch1/invalidation-2.mlir:106:18: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
                 ^
test/Examples/transform/Ch1/invalidation-2.mlir:24:13: note: ancestor payload op
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
test/Examples/transform/Ch1/invalidation-2.mlir:24:13: note: nested payload op
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
请注意，“添加”元素操作被指示为有效载荷祖先，因为它用于生成图块循环，因此循环有其位置。

最后，我们想用对微内核的调用来替换对上述函数的调用。不幸的是，Transform 方言不支持这种转换（如果将调用重写为自定义的树外操作，则无法支持）。因此，我们需要定义新的转换操作。下一章将介绍如何做到这一点。
```

## 跟踪 IR 修改 
```
变换方言会自动跟踪作为变换操作的一部分所做的所有 IR 更改。（实现必须使用提供的重写器来修改 IR。）如果有效载荷操作被删除，它将自动从当前与之关联的所有句柄中删除。如果有效载荷操作被替换，变换方言会尝试找到替换操作并相应地更新所有句柄。如果将多结果操作替换为由多个操作定义的值，或者将操作替换为不同类型的操作，则会产生错误。这是因为不清楚直接替换是否真正代表了原始操作的计算。有多种方法可以自定义此行为。更多详细信息请参阅的文档transform::TrackingListener。
```
