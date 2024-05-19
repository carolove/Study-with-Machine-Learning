# 用mlir代码理解机器编译之matmul从linalg和vector降级lowering到设备级IR
- linalg
- vector
- Unrolling
- 清理高维向量
- Hoisting
- Lowering
- 每一步都对比代码，然后和原文注解进行对比，争取理解 affine_map
## linalg 层级的matmul
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c64 step %7 {
      %8 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %8) -> (tensor<8x32xf32>) {
        %13 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %14 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %15 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [4, 4], [0, 0, 4]]>} ins(%cst : f32) outs(%15 : tensor<4x4xf32>) -> tensor<4x4xf32>
          %17 = tensor.extract_slice %9[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
          %18 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %19 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %16) -> (tensor<4x4xf32>) {
            %22 = tensor.extract_slice %17[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %23 = tensor.extract_slice %18[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %24 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [4, 4], [0, 0, 4]]>} ins(%22, %23 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg7 : tensor<4x4xf32>) -> tensor<4x4xf32>
            scf.yield %24 : tensor<4x4xf32>
          }
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<4x4xf32>) outs(%19 : tensor<4x4xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 32], [4, 4], [0, 0, 4]]>} {
          ^bb0(%arg6: f32, %arg7: f32):
            %22 = arith.subf %arg7, %arg6 : f32
            linalg.yield %22 : f32
          } -> tensor<4x4xf32>
          %21 = tensor.insert_slice %20 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %21 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %13 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```

## vector层级的matmul
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %cst = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c64 step %7 {
      %8 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %8) -> (tensor<8x32xf32>) {
        %13 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %14 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %15 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = vector.transfer_write %cst, %15[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
          %17 = tensor.extract_slice %9[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
          %18 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %19 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %16) -> (tensor<4x4xf32>) {
            %25 = tensor.extract_slice %17[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %26 = tensor.extract_slice %18[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %27 = vector.transfer_read %25[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
            %28 = vector.transfer_read %26[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
            %29 = vector.transfer_read %arg7[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
            %30 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %28, %29 : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
            %31 = vector.transfer_write %30, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
            scf.yield %31 : tensor<4x4xf32>
          }
          %20 = vector.transfer_read %14[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
          %21 = vector.transfer_read %19[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
          %22 = arith.subf %21, %20 : vector<4x4xf32>
          %23 = vector.transfer_write %22, %19[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
          %24 = tensor.insert_slice %23 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %24 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %13 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```

## Unrolling展开
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c64 step %7 {
      %8 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %8) -> (tensor<8x32xf32>) {
        %13 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %14 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %15 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = vector.extract_strided_slice %cst {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xf32> to vector<1x4xf32>
          %17 = vector.transfer_write %16, %15[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %18 = vector.extract_strided_slice %cst {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xf32> to vector<1x4xf32>
          %19 = vector.transfer_write %18, %17[%c1, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %20 = vector.extract_strided_slice %cst {offsets = [2, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xf32> to vector<1x4xf32>
          %21 = vector.transfer_write %20, %19[%c2, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %22 = vector.extract_strided_slice %cst {offsets = [3, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xf32> to vector<1x4xf32>
          %23 = vector.transfer_write %22, %21[%c3, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %24 = tensor.extract_slice %9[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
          %25 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %26 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %23) -> (tensor<4x4xf32>) {
            %44 = tensor.extract_slice %24[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %45 = tensor.extract_slice %25[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %46 = vector.transfer_read %44[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %47 = vector.transfer_read %44[%c1, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %48 = vector.transfer_read %44[%c2, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %49 = vector.transfer_read %44[%c3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %50 = vector.transfer_read %45[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %51 = vector.transfer_read %45[%c1, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %52 = vector.transfer_read %45[%c2, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %53 = vector.transfer_read %45[%c3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %54 = vector.transfer_read %arg7[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %55 = vector.transfer_read %arg7[%c1, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %56 = vector.transfer_read %arg7[%c2, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %57 = vector.transfer_read %arg7[%c3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
            %58 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %59 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %58, %50, %54 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %60 = vector.extract_strided_slice %46 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %61 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %60, %51, %59 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %62 = vector.extract_strided_slice %46 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %63 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %62, %52, %61 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %64 = vector.extract_strided_slice %46 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %65 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %64, %53, %63 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %66 = vector.extract_strided_slice %47 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %67 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %66, %50, %55 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %68 = vector.extract_strided_slice %47 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %69 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %68, %51, %67 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %70 = vector.extract_strided_slice %47 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %71 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %70, %52, %69 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %72 = vector.extract_strided_slice %47 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %73 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %72, %53, %71 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %74 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %75 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %74, %50, %56 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %76 = vector.extract_strided_slice %48 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %77 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %76, %51, %75 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %78 = vector.extract_strided_slice %48 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %79 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %78, %52, %77 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %80 = vector.extract_strided_slice %48 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %81 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %80, %53, %79 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %82 = vector.extract_strided_slice %49 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %83 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %82, %50, %57 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %84 = vector.extract_strided_slice %49 {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %85 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %84, %51, %83 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %86 = vector.extract_strided_slice %49 {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %87 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %86, %52, %85 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %88 = vector.extract_strided_slice %49 {offsets = [0, 3], sizes = [1, 1], strides = [1, 1]} : vector<1x4xf32> to vector<1x1xf32>
            %89 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %88, %53, %87 : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
            %90 = vector.transfer_write %65, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
            %91 = vector.transfer_write %73, %90[%c1, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
            %92 = vector.transfer_write %81, %91[%c2, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
            %93 = vector.transfer_write %89, %92[%c3, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
            scf.yield %93 : tensor<4x4xf32>
          }
          %27 = vector.transfer_read %14[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %28 = vector.transfer_read %14[%c1, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %29 = vector.transfer_read %14[%c2, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %30 = vector.transfer_read %14[%c3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %31 = vector.transfer_read %26[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %32 = vector.transfer_read %26[%c1, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %33 = vector.transfer_read %26[%c2, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %34 = vector.transfer_read %26[%c3, %c0], %cst_0 {in_bounds = [true, true]} : tensor<4x4xf32>, vector<1x4xf32>
          %35 = arith.subf %31, %27 : vector<1x4xf32>
          %36 = arith.subf %32, %28 : vector<1x4xf32>
          %37 = arith.subf %33, %29 : vector<1x4xf32>
          %38 = arith.subf %34, %30 : vector<1x4xf32>
          %39 = vector.transfer_write %35, %26[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %40 = vector.transfer_write %36, %39[%c1, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %41 = vector.transfer_write %37, %40[%c2, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %42 = vector.transfer_write %38, %41[%c3, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<4x4xf32>
          %43 = tensor.insert_slice %42 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %43 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %13 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```
## 清理高维向量
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c64 step %7 {
      %8 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %8) -> (tensor<8x32xf32>) {
        %13 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %14 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %15 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = vector.transfer_write %cst, %15[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %17 = vector.transfer_write %cst, %16[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %18 = vector.transfer_write %cst, %17[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %19 = vector.transfer_write %cst, %18[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %20 = tensor.extract_slice %9[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
          %21 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %22 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %19) -> (tensor<4x4xf32>) {
            %40 = tensor.extract_slice %20[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %41 = tensor.extract_slice %21[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %42 = vector.transfer_read %40[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %43 = vector.transfer_read %40[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %44 = vector.transfer_read %40[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %45 = vector.transfer_read %40[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %46 = vector.transfer_read %41[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %47 = vector.broadcast %46 : vector<4xf32> to vector<1x4xf32>
            %48 = vector.transfer_read %41[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %49 = vector.broadcast %48 : vector<4xf32> to vector<1x4xf32>
            %50 = vector.transfer_read %41[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %51 = vector.broadcast %50 : vector<4xf32> to vector<1x4xf32>
            %52 = vector.transfer_read %41[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %53 = vector.broadcast %52 : vector<4xf32> to vector<1x4xf32>
            %54 = vector.transfer_read %arg7[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %55 = vector.transfer_read %arg7[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %56 = vector.transfer_read %arg7[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %57 = vector.transfer_read %arg7[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %58 = vector.extract_strided_slice %42 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %59 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %58, %47, %54 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %60 = vector.extract_strided_slice %42 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %61 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %60, %49, %59 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %62 = vector.extract_strided_slice %42 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %63 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %62, %51, %61 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %64 = vector.extract_strided_slice %42 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %65 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %64, %53, %63 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %66 = vector.extract_strided_slice %43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %67 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %66, %47, %55 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %68 = vector.extract_strided_slice %43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %69 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %68, %49, %67 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %70 = vector.extract_strided_slice %43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %71 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %70, %51, %69 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %72 = vector.extract_strided_slice %43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %73 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %72, %53, %71 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %74 = vector.extract_strided_slice %44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %75 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %74, %47, %56 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %76 = vector.extract_strided_slice %44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %77 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %76, %49, %75 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %78 = vector.extract_strided_slice %44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %79 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %78, %51, %77 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %80 = vector.extract_strided_slice %44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %81 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %80, %53, %79 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %82 = vector.extract_strided_slice %45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %83 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %82, %47, %57 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %84 = vector.extract_strided_slice %45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %85 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %84, %49, %83 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %86 = vector.extract_strided_slice %45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %87 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %86, %51, %85 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %88 = vector.extract_strided_slice %45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %89 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %88, %53, %87 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %90 = vector.transfer_write %65, %arg7[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
            %91 = vector.transfer_write %73, %90[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
            %92 = vector.transfer_write %81, %91[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
            %93 = vector.transfer_write %89, %92[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
            scf.yield %93 : tensor<4x4xf32>
          }
          %23 = vector.transfer_read %14[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %24 = vector.transfer_read %14[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %25 = vector.transfer_read %14[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %26 = vector.transfer_read %14[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %27 = vector.transfer_read %22[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %28 = vector.transfer_read %22[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %29 = vector.transfer_read %22[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %30 = vector.transfer_read %22[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %31 = arith.subf %27, %23 : vector<4xf32>
          %32 = arith.subf %28, %24 : vector<4xf32>
          %33 = arith.subf %29, %25 : vector<4xf32>
          %34 = arith.subf %30, %26 : vector<4xf32>
          %35 = vector.transfer_write %31, %22[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %36 = vector.transfer_write %32, %35[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %37 = vector.transfer_write %33, %36[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %38 = vector.transfer_write %34, %37[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %39 = tensor.insert_slice %38 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %39 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %13 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```
## Hoisting
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %c128 step %5 {
    %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
    scf.for %arg1 = %6 to %c64 step %7 {
      %9 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %9) -> (tensor<8x32xf32>) {
        %13 = tensor.extract_slice %8[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
        %14 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %15 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %17 = vector.transfer_write %cst, %16[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %18 = vector.transfer_write %cst, %17[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %19 = vector.transfer_write %cst, %18[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %20 = vector.transfer_write %cst, %19[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %21 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %22:4 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
            %40 = tensor.extract_slice %13[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %41 = tensor.extract_slice %21[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %42 = vector.transfer_read %40[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %43 = vector.transfer_read %40[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %44 = vector.transfer_read %40[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %45 = vector.transfer_read %40[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %46 = vector.transfer_read %41[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %47 = vector.broadcast %46 : vector<4xf32> to vector<1x4xf32>
            %48 = vector.transfer_read %41[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %49 = vector.broadcast %48 : vector<4xf32> to vector<1x4xf32>
            %50 = vector.transfer_read %41[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %51 = vector.broadcast %50 : vector<4xf32> to vector<1x4xf32>
            %52 = vector.transfer_read %41[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %53 = vector.broadcast %52 : vector<4xf32> to vector<1x4xf32>
            %54 = vector.extract_strided_slice %42 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %55 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %54, %47, %arg10 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %56 = vector.extract_strided_slice %42 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %57 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %56, %49, %55 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %58 = vector.extract_strided_slice %42 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %59 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %58, %51, %57 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %60 = vector.extract_strided_slice %42 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %61 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %60, %53, %59 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %62 = vector.extract_strided_slice %43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %63 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %62, %47, %arg9 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %64 = vector.extract_strided_slice %43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %65 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %64, %49, %63 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %66 = vector.extract_strided_slice %43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %67 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %66, %51, %65 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %68 = vector.extract_strided_slice %43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %69 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %68, %53, %67 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %70 = vector.extract_strided_slice %44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %71 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %70, %47, %arg8 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %72 = vector.extract_strided_slice %44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %73 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %72, %49, %71 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %74 = vector.extract_strided_slice %44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %75 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %74, %51, %73 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %76 = vector.extract_strided_slice %44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %77 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %76, %53, %75 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %78 = vector.extract_strided_slice %45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %79 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %78, %47, %arg7 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %80 = vector.extract_strided_slice %45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %81 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %80, %49, %79 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %82 = vector.extract_strided_slice %45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %83 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %82, %51, %81 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            %84 = vector.extract_strided_slice %45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
            %85 = vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>} %84, %53, %83 : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
            scf.yield %85, %77, %69, %61 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }
          %23 = vector.transfer_write %22#3, %20[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %24 = vector.transfer_write %22#2, %23[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %25 = vector.transfer_write %22#1, %24[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %26 = vector.transfer_write %22#0, %25[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %27 = vector.transfer_read %15[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %28 = vector.transfer_read %15[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %29 = vector.transfer_read %15[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %30 = vector.transfer_read %15[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %31 = arith.subf %22#3, %27 : vector<4xf32>
          %32 = arith.subf %22#2, %28 : vector<4xf32>
          %33 = arith.subf %22#1, %29 : vector<4xf32>
          %34 = arith.subf %22#0, %30 : vector<4xf32>
          %35 = vector.transfer_write %31, %26[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %36 = vector.transfer_write %32, %35[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %37 = vector.transfer_write %33, %36[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %38 = vector.transfer_write %34, %37[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %39 = tensor.insert_slice %38 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %39 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %14 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```
## Lowering
```
func.func @dot_dispatch_0_matmul_128x64x256() {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x256xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:256x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x64xf32>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %c128 step %5 {
    %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<8x256xf32>
    scf.for %arg1 = %6 to %c64 step %7 {
      %9 = flow.dispatch.tensor.load %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x64xf32> -> tensor<8x32xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x64xf32> -> tensor<256x32xf32>
      %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x64xf32> -> tensor<8x32xf32>
      %12 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %9) -> (tensor<8x32xf32>) {
        %13 = tensor.extract_slice %8[%arg2, 0] [4, 256] [1, 1] : tensor<8x256xf32> to tensor<4x256xf32>
        %14 = scf.for %arg4 = %c0 to %c32 step %c4 iter_args(%arg5 = %arg3) -> (tensor<8x32xf32>) {
          %15 = tensor.extract_slice %11[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %16 = tensor.extract_slice %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
          %17 = tensor.extract_slice %10[0, %arg4] [256, 4] [1, 1] : tensor<256x32xf32> to tensor<256x4xf32>
          %18:4 = scf.for %arg6 = %c0 to %c256 step %c4 iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
            %32 = tensor.extract_slice %13[0, %arg6] [4, 4] [1, 1] : tensor<4x256xf32> to tensor<4x4xf32>
            %33 = tensor.extract_slice %17[%arg6, 0] [4, 4] [1, 1] : tensor<256x4xf32> to tensor<4x4xf32>
            %34 = vector.transfer_read %32[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %35 = vector.transfer_read %32[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %36 = vector.transfer_read %32[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %37 = vector.transfer_read %32[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %38 = vector.transfer_read %33[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %39 = vector.transfer_read %33[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %40 = vector.transfer_read %33[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %41 = vector.transfer_read %33[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
            %42 = vector.extract %34[0] : vector<4xf32>
            %43 = vector.splat %42 : vector<4xf32>
            %44 = vector.fma %43, %38, %arg10 : vector<4xf32>
            %45 = vector.extract %34[1] : vector<4xf32>
            %46 = vector.splat %45 : vector<4xf32>
            %47 = vector.fma %46, %39, %44 : vector<4xf32>
            %48 = vector.extract %34[2] : vector<4xf32>
            %49 = vector.splat %48 : vector<4xf32>
            %50 = vector.fma %49, %40, %47 : vector<4xf32>
            %51 = vector.extract %34[3] : vector<4xf32>
            %52 = vector.splat %51 : vector<4xf32>
            %53 = vector.fma %52, %41, %50 : vector<4xf32>
            %54 = vector.extract %35[0] : vector<4xf32>
            %55 = vector.splat %54 : vector<4xf32>
            %56 = vector.fma %55, %38, %arg9 : vector<4xf32>
            %57 = vector.extract %35[1] : vector<4xf32>
            %58 = vector.splat %57 : vector<4xf32>
            %59 = vector.fma %58, %39, %56 : vector<4xf32>
            %60 = vector.extract %35[2] : vector<4xf32>
            %61 = vector.splat %60 : vector<4xf32>
            %62 = vector.fma %61, %40, %59 : vector<4xf32>
            %63 = vector.extract %35[3] : vector<4xf32>
            %64 = vector.splat %63 : vector<4xf32>
            %65 = vector.fma %64, %41, %62 : vector<4xf32>
            %66 = vector.extract %36[0] : vector<4xf32>
            %67 = vector.splat %66 : vector<4xf32>
            %68 = vector.fma %67, %38, %arg8 : vector<4xf32>
            %69 = vector.extract %36[1] : vector<4xf32>
            %70 = vector.splat %69 : vector<4xf32>
            %71 = vector.fma %70, %39, %68 : vector<4xf32>
            %72 = vector.extract %36[2] : vector<4xf32>
            %73 = vector.splat %72 : vector<4xf32>
            %74 = vector.fma %73, %40, %71 : vector<4xf32>
            %75 = vector.extract %36[3] : vector<4xf32>
            %76 = vector.splat %75 : vector<4xf32>
            %77 = vector.fma %76, %41, %74 : vector<4xf32>
            %78 = vector.extract %37[0] : vector<4xf32>
            %79 = vector.splat %78 : vector<4xf32>
            %80 = vector.fma %79, %38, %arg7 : vector<4xf32>
            %81 = vector.extract %37[1] : vector<4xf32>
            %82 = vector.splat %81 : vector<4xf32>
            %83 = vector.fma %82, %39, %80 : vector<4xf32>
            %84 = vector.extract %37[2] : vector<4xf32>
            %85 = vector.splat %84 : vector<4xf32>
            %86 = vector.fma %85, %40, %83 : vector<4xf32>
            %87 = vector.extract %37[3] : vector<4xf32>
            %88 = vector.splat %87 : vector<4xf32>
            %89 = vector.fma %88, %41, %86 : vector<4xf32>
            scf.yield %89, %77, %65, %53 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }
          %19 = vector.transfer_read %15[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %20 = vector.transfer_read %15[%c1, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %21 = vector.transfer_read %15[%c2, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %22 = vector.transfer_read %15[%c3, %c0], %cst_0 {in_bounds = [true]} : tensor<4x4xf32>, vector<4xf32>
          %23 = arith.subf %18#3, %19 : vector<4xf32>
          %24 = arith.subf %18#2, %20 : vector<4xf32>
          %25 = arith.subf %18#1, %21 : vector<4xf32>
          %26 = arith.subf %18#0, %22 : vector<4xf32>
          %27 = vector.transfer_write %23, %16[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %28 = vector.transfer_write %24, %27[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %29 = vector.transfer_write %25, %28[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %30 = vector.transfer_write %26, %29[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4xf32>
          %31 = tensor.insert_slice %30 into %arg5[%arg2, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
          scf.yield %31 : tensor<8x32xf32>
        } {iree.spirv.distribute_dim = 0 : index}
        scf.yield %14 : tensor<8x32xf32>
      } {iree.spirv.distribute_dim = 1 : index}
      flow.dispatch.tensor.store %12, %3, offsets = [%arg0, %arg1], sizes = [8, 32], strides = [1, 1] : tensor<8x32xf32> -> !flow.dispatch.tensor<writeonly:128x64xf32>
    }
  }
  return
}
```
