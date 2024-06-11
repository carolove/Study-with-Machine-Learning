// ../../../../../build/bin/mlir-opt sequence-03.mlir --pass-pipeline="builtin.module(test-transform-dialect-interpreter{  bind-first-extra-to-ops=linalg.matmul   bind-second-extra-to-ops=linalg.elemwise_binary   enable-expensive-checks})"

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
