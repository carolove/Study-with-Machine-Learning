//  ../../../../../build/bin/mlir-opt sequence-00.mlir --pass-pipeline="builtin.module(test-transform-dialect-interpreter{  bind-first-extra-to-ops=linalg.matmul bind-second-extra-to-ops=linalg.elemwise_binary})"

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
  transform.debug.emit_remark_at %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.debug.emit_remark_at %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
