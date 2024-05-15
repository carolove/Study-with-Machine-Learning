# 用mlir代码理解机器编译之matmul从linalg到vector的lowering
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
