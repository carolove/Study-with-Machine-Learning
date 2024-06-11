//  ../../../../../build/bin/mlir-opt sequence-02.mlir --pass-pipeline="builtin.module(test-transform-dialect-interpreter{  bind-first-extra-to-ops=linalg.matmul   bind-second-extra-to-ops=linalg.elemwise_binary   enable-expensive-checks})"

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

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // We can cast one type to another as long as operations are compatible
  // with both types. This creates "aliasing" handles.
  %casted = transform.cast %arg1 : !transform.op<"linalg.matmul">
      to !transform.any_op

  // The actual tiling transformation takes tile sizes as attributes.
  %loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
    : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

  // Consuming an operand invalidates the consumed handle and any other handle that is
  // associated with the same payload operations, or payload operations nested in them.
  // 尽管使用了别名，但是仍然会出现引用失效，检查可以规避这种情况
  transform.debug.emit_remark_at %casted, "remark"
    : !transform.any_op
  transform.yield
}
