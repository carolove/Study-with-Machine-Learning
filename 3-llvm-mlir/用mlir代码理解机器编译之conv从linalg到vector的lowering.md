# 用mlir代码理解机器编译之conv从linalg到vector的lowering

## linalg 层级的conv
```
%26 = tensor.extract_slice ...
%27 = tensor.extract_slice %arg6...
%28 = linalg.fill {...} ins(%cst : f32) outs(%27 : tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xf32>
%35 = tensor.extract_slice ...
%36 = tensor.extract_slice ...
%37 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %28) -> (tensor<1x1x2x4xf32>) {
  %40 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (tensor<1x1x2x4xf32>) {
    %49 = tensor.extract_slice ...
    %50 = tensor.pad %49 low[0, 0, 0, 0] high[0, %44, %48, 0] {
    ^bb0(%arg11: index, %arg12: index, %arg13: index, %arg14: index):
      tensor.yield %cst : f32
    } : tensor<1x?x?x3xf32> to tensor<1x1x3x3xf32>
    %51 = tensor.extract_slice ...
    %52 = linalg.conv_2d_nhwc_hwcf
          {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
          ins(%50, %51 : tensor<1x1x3x3xf32>, tensor<1x1x3x4xf32>)
          outs(%arg10 : tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xf32>
    scf.yield %52 : tensor<1x1x2x4xf32>
  }
  scf.yield %40 : tensor<1x1x2x4xf32>
}
%38 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
} ins(%26 : tensor<1x1x2x4xf32>) outs(%37 : tensor<1x1x2x4xf32>) attrs =  {...} {
^bb0(%arg7: f32, %arg8: f32):
  %40 = arith.subf %arg8, %arg7 : f32
  linalg.yield %40 : f32
} -> tensor<1x1x2x4xf32>
%39 = tensor.insert_slice %38 into %arg6...
```

## vector层级的conv
```
%26 = tensor.extract_slice ...
%27 = tensor.extract_slice %arg6...
%28 = vector.transfer_write %cst, %27[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf32>, tensor<1x1x2x4xf32>
%35 = tensor.extract_slice ...
%36 = tensor.extract_slice ...
%37 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %28) -> (tensor<1x1x2x4xf32>) {
  %43 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %arg8) -> (tensor<1x1x2x4xf32>) {
    %50 = tensor.extract_slice ...
    %56 = scf.if ... -> (vector<3xf32>) {
      %93 = vector.transfer_read %50[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
      scf.yield %93 : vector<3xf32>
    } else {
      scf.yield %cst_1 : vector<3xf32>
    }
    %57 = vector.insert_strided_slice %56, %cst_0 {offsets = [0, 0], strides = [1]} : vector<3xf32> into vector<3x3xf32>
    %61 = scf.if ... -> (vector<3xf32>) {
      %93 = vector.transfer_read %50[%c0, %c0, %c1, %c0], %cst_2 {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
      scf.yield %93 : vector<3xf32>
    } else {
      scf.yield %cst_1 : vector<3xf32>
    }
    %62 = vector.insert_strided_slice %61, %57 {offsets = [1, 0], strides = [1]} : vector<3xf32> into vector<3x3xf32>
    %66 = scf.if ... -> (vector<3xf32>) {
      %93 = vector.transfer_read %50[%c0, %c0, %c2, %c0], %cst_2 {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
      scf.yield %93 : vector<3xf32>
    } else {
      scf.yield %cst_1 : vector<3xf32>
    }
    %67 = vector.insert_strided_slice %66, %62 {offsets = [2, 0], strides = [1]} : vector<3xf32> into vector<3x3xf32>
    %68 = linalg.init_tensor [1, 1, 3, 3] : tensor<1x1x3x3xf32>
    %69 = vector.transfer_write %67, %68[%c0, %c0, %c0, %c0] {in_bounds = [true, true]} : vector<3x3xf32>, tensor<1x1x3x3xf32>
    %70 = tensor.extract_slice %36[%arg7, %arg9, 0, 0] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x4xf32> to tensor<1x1x3x4xf32>
    %71 = vector.transfer_read %70[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x1x3x4xf32>, vector<3x4xf32>
    %72 = vector.extract_strided_slice %71 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]} : vector<3x4xf32> to vector<1x4xf32>
    %73 = vector.extract_strided_slice %71 {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<3x4xf32> to vector<1x4xf32>
    %74 = vector.extract_strided_slice %71 {offsets = [2, 0], sizes = [1, 4], strides = [1, 1]} : vector<3x4xf32> to vector<1x4xf32>
    %75 = vector.transfer_read %69[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x1x3x3xf32>, vector<1x3xf32>
    %76 = vector.transfer_read %arg10[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x1x2x4xf32>, vector<1x4xf32>
    %77 = vector.extract_strided_slice %75 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %78 = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"],
            kind = #vector.kind<add>
          } %77, %72, %76 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %79 = vector.extract_strided_slice %75 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %80 = vector.contract {...} %79, %73, %78 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %81 = vector.extract_strided_slice %75 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %82 = vector.contract {...} %81, %74, %80 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %83 = vector.transfer_write %82, %arg10[%c0, %c0, %c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x1x2x4xf32>
    %84 = vector.transfer_read %69[%c0, %c0, %c2, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x1x3x3xf32>, vector<1x3xf32>
    %85 = vector.transfer_read %arg10[%c0, %c0, %c1, %c0], %cst_2 {in_bounds = [true, true]} : tensor<1x1x2x4xf32>, vector<1x4xf32>
    %86 = vector.extract_strided_slice %84 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %87 = vector.contract {...} %86, %72, %85 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %88 = vector.extract_strided_slice %84 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %89 = vector.contract {...} %88, %73, %87 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %90 = vector.extract_strided_slice %84 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x3xf32> to vector<1x1xf32>
    %91 = vector.contract {...} %90, %74, %89 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
    %92 = vector.transfer_write %91, %83[%c0, %c0, %c1, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x1x2x4xf32>
    scf.yield %92 : tensor<1x1x2x4xf32>
  }
  scf.yield %43 : tensor<1x1x2x4xf32>
}
%38 = vector.transfer_read %26[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true, true, true, true]} : tensor<1x1x2x4xf32>, vector<1x1x2x4xf32>
%39 = vector.transfer_read %37[%c0, %c0, %c0, %c0], %cst_2 {in_bounds = [true, true, true, true]} : tensor<1x1x2x4xf32>, vector<1x1x2x4xf32>
%40 = arith.subf %39, %38 : vector<1x1x2x4xf32>
%41 = vector.transfer_write %40, %37[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x4xf32>, tensor<1x1x2x4xf32>
%42 = tensor.insert_slice %41 into %arg6...
```
