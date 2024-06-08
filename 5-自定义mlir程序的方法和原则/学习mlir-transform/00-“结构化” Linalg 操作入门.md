# 第 0 章："结构化" Linalg 操作入门
- 在开始 Transform 方言教程之前，让我们简单了解一下结构化操作的概念及其在 Linalg 方言中的实现。请注意，Transform 方言不需要结构化操作，反之亦然。两者在 Transform 方言开始时共同发展，这使得结构化操作的转换子集最成熟，最适合本教程。如果您已经熟悉这个概念，请跳至第 1 章。
- 结构化代码生成旨在尽可能长时间地保留计算结构以支持转换，直至支持特定转换的 IR 抽象的设计。

## 统一元素扩展 
```
考虑 MLIR 中的简单标量算术加法运算，它直接映射到大多数支持浮点运算的架构上的机器指令：

%2 = arith.addf %0, %1 : f32
这个操作可以很容易地扩展以统一应用于一维向量的元素，这也经常作为向量机的指令使用：

%2 = arith.addf %0, %1 : vector<8xf32>
只有少数现代指令集提供针对二维或多维向量的指令。然而，在 MLIR 中，可以透明地将统一元素应用扩展到任意阶向量。

%2 = arith.addf %0, %1 : vector<8x4xf32>
%5 = arith.addf %3, %4 : vector<2x2x2x2x2x2x2xf32>
您可以注意到，MLIR 对向量的算术运算保留了统一元素应用的结构。例如，编译器可以利用此结构来生成目标上可用的较小等级的运算，或者在有此类融合指令可用时融合乘法和加法（当有一百次乘法和一百次加法时，这会变得复杂）。
```
## 缩减 
```
有时需要将向量的元素相加以获得标量。有些平台提供了针对此操作的特定指令，有些平台提供了可以组合以实现所需效果的指令，例如相邻元素的相加和元素混洗。

MLIR 中的向量方言定义了一个操作来明确表示向量内缩减：

%0 = vector.reduction <add>, %0 : vector<8xf32> into f32
当没有支持时，这样的操作可以转变为循环：

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c8 = arith.constant 8 : index
%init = arith.constant 0.0 : f32
%result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
  %element = vector.extractelement %0[%i : index] : vector<8xf32>
  %updated = arith.addf %partial, %element : f32
  scf.yield %updated : f32
}
即使有特殊指令可用，根据指令延迟和寄存器压力，仍可能需要使用循环形式（带展开）。将操作结构保留为单个归约可以让编译器了解执行的是向量内归约，因此可以选择实现。
```
## 收缩 
```
收缩是归约的一种推广，它将两个向量中的元素相乘，然后再相加。简单的“加”归约可以看作是收缩，其中一个向量包含1.0乘法的中性元素。收缩为编译器提供了更大的灵活性，并由 MLIR 中的专用操作表示：

// Neutral initializer for the addition.
%init  = arith.constant 0.0 : f32
// Neutral element of multiplication.
%ones = arith.constant dense<1.0> : vector<8xf32>
// Actual contraction.
%result = vector.contract {
  indexing_maps = [affine_map<(i) -> (i)>,
                   affine_map<(i) -> (i)>,
                   affine_map<(i) -> ()>],
  iterator_types = ["reduction"]
} %0, %ones, %init : vector<8xf32>, vector<8xf32> into f32
注意affine_map是表达如何对向量元素进行索引的表达式。可以用下面的伪代码来描述收缩模式的等效循环形式：

for i in 0 to 8:
  init += p0[i] * ones[i]
如相应仿射映射所示(i) -> (i)，代表着其中%0和都%ones是以此读取向量元素；而(i) -> ()，意味着 %init 他是收缩/缩减/降维的。

类似的，MLIR 向量收缩不仅限于 1D 情况。在 2D+ 情况下，还可以指定哪些向量维度正在缩减以及哪些维度正在保留。这可以通过使用属性来实现iterator_types，该属性为每个维度指定是缩减 ( "reduction") 还是保留 ( "parallel")。考虑以下编码矩阵-矩阵乘法的 3D 收缩：

%result = vector.contract {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
查看索引图，很容易识别出循环形式：

for i in 0 to 8:
  for j in 0 to 16:
    for k in 0 to 10:
      init[i, j] += lhs[i, k] * rhs[k, j]
保留这种收缩的高级结构使得编译器更容易识别矩阵乘法和点积等运算，并使其可以自由地生成利用大多数高级指令甚至预生成的微内核的低级运算。
```
## 内存通用操作 
```
到目前为止，我们一直在考虑对存储在虚拟寄存器中的向量进行操作。可以在内存中定义类似的收缩抽象：

linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : memref<8x10xf32>, memref<10x16xf32>)
  outs(%init : memref<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
}
这看起来更复杂，所以让我们来分析一下。indexing_maps和与我们上面看到的向量收缩完全相同iterator_types。操作数现在分为两个列表：

in包含仅由操作读取的缓冲区的操作数；
out正在由操作读取和更新的操作数。
这种分离对于向量来说不是必要的，因为在 MLIR 中，向量是只读的（SSA 或功能形式），而改变向量的操作实际上是会产生一个新的向量。

此外，该操作现在包含一个区域，该区域明确指定了收缩中隐含的乘法和加法运算。区域中的块参数对应于从缓冲区读取的各个元素：前两个对应于操作in数，最后一个对应于操作out数。从该区域产生的值被“写入”到操作out数，并可用作该区域未来执行的最后一个块参数。请注意，对于从缓冲区读取的各种元素元组，区域执行的顺序没有指定，并且out在操作结束时将写入缓冲区作为一个整体。
```
## "Loop" 融合Fusion 
```
由于操作区域linalg.generic可以包含任意多个操作，我们可以用它来表达隐式循环的“融合”，只需在区域中链接更多操作即可。例如，常见的机器学习整流线性单元层（ReLU）可以定义为，relu(x) = max(0, x)可以在一个操作中使用“比较和选择”习语来定义linalg.generic，而无需比较结果的临时缓冲区，也无需重复外部操作：

linalg.generic {
  indexing_maps [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
  iterator_types = ["parallel"]
} ins(%in : memref<?xf32>) outs(%out : memref<?xf32>) {
^bb0(%in_one : f32, %out_one : f32):
  %c0 = arith.constant 0.0 : f32
  %0 = arith.cmpf ogt %in_one, %c0 : f32
  %1 = arith.select %0, %in_one, %c0 : f32
  linalg.yield %1 : f32 
}
此类操作可以转换为循环，也可以拆分为多个操作后转换为向量形式，每个操作都映射到一个向量方言原语。这种建模再次为编译器提供了更多选择，可以选择代码生成策略。
```
## 张量的通用操作 
```
让我们在抽象阶梯上再迈出最后一步。MLIR 提供了张量抽象，使编译器可以轻松推理多维但规则的数据，而无需解决多维缓冲区所必需的复杂问题，例如别名分析和依赖项满足。张量抽象与向量抽象非常相似（主要区别包括无序张量的可用性、张量布局以及向量可用作张量的元素类型，但不能用作其他向量）。张量是只读的，更新张量的操作会产生新的张量。

linalg.generic上面的操作可以提升到对张量而不是缓冲区进行操作：

%result = linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : tensor<8x10xf32>,tensor<10x16xf32>)
  outs(%init :tensor<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
} -> tensor<8x16xf32>
您可以注意到，此操作的大多数组件与其缓冲区版本相同。它是专门以这种方式设计的。除了操作数类型之外，主要区别在于该操作现在会产生新结果，而不是更新缓冲区out。out操作数仅用作初始化值。

如果该linalg.generic操作存在于向量上，它将具有完全相同的结构。
```
## 平铺Tiling和循环实现 
```
在这个抽象级别上，编译器可以轻松执行通常需要高性能代码生成的更高级的转换，例如 平铺Tiling。通常，平铺Tiling可以看作是将迭代空间划分为更小的部分或图块，以便每个部分所需的数据适合缓存级别。执行图块的顺序必须保留原始数据依赖关系。

对于linalg.generic操作，迭代空间是隐式的，由操作数的形状定义。因此，可以通过对原始数据的子集（切片）执行相同的操作来表示图块。由于将主体linalg.generic应用于输入元素的不同元组的顺序未指定，因此可以按任何顺序执行图块，而无需进行依赖性分析。为了控制不同图块的执行，图块的实现会产生循环。因此，图块linalg.generic操作也可以看作是实现迄今为止隐式的循环。

例如，将上面展示的矩阵乘法用平铺Tiling大小进行平铺Tiling(2, 8)，我们得到一个嵌套的循环，围绕linalg.generic一个张量表达相同的操作2x8。

// A special "multi-for" loop that supports tensor-insertion semantics 
// as opposed to implicit updates. The resulting 8x16 tensor will be produced
// by this loop.
// The trip count of iterators is computed dividing the original tensor size,
// 8x16, by the tile size, 2x8, to obtain 4x2.
// When tensor sizes are dynamic, the trip count computation is emitted as IR
// and is being computed at runtime.
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {

  // Scale the loop induction variables by the tile sizes.
  %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%3, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %4] [10, 8] [1, 1] 
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %shared[%3, %4] [2, 8] [1, 1] 
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same operation as before, but now operating on smaller
  // slices of data.
  %partial =  linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>) 
    outs(%result_slice : tensor<2x8xf32>) -> tensor<2x8xf32> {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } : tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial into %shared[%3, %4] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```
## 生产者Producer / 消费者Consumer融合与重新实现 
```
在使用平铺Tiling实现循环之后，另一个关键的代码生成转换变得简单——融合。与循环融合不同，结构化操作方法允许生产者Producer/消费者融合，即使操作的（隐式）迭代空间不匹配。给定张量的高级结构化操作，例如linalg.generic，可以按照 use-def 链来识别：

该图块使用的操作数的子集（切片），以及
产生被切片的整个张量的张量级结构化操作。
通过反转indexing_map并将其应用于通过切片访问的元素集，我们可以计算定义计算图块所需的完整张量的操作的迭代空间部分。因此，融合归结为用产生原始操作数tensor.extract_slice的图块替换操作。linalg.generic

假设矩阵乘法运算后跟有另一个运算，该运算将结果矩阵的每个元素与其自身相乘。此尾随元素运算具有 2D 迭代空间，与矩阵乘法中的 3D 迭代空间不同。尽管如此，仍可以平铺Tiling尾随运算，然后将其操作数的生成者 matmul 融合到平铺Tiling生成的循环中。未平铺Tiling的维度将被全部使用。

// Same loop as before.
%0 = scf.forall (%i, %j) in (4, 2) 
     shared_outs(%shared = %init) 
     -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // Scale the loop induction variables by the tile sizes.
  %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%1, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %2] [10, 8] [1, 1]
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %result[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same matmul slice as before. It replaces the slice
  // extraction for the generic operation below.
  %partial = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>)
   outs(%result_slice : tensor<2x8xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %5 = arith.mulf %lhs_one, %rhs_one : f32
    %6 = arith.addf %init_one, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2x8xf32>

  // Take the slice of the final result. Note that we don't need to take
  // the slice of the operand because the matmul operation above computes
  // it in-place.
  %shared_slice = tensor.extract_slice %shared[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // The elementwise operation that we tiled.
  %elemwise = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%partial : tensor<2x8xf32>)   
   outs(%shared_slice : tensor<2x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.mulf %in, %in : f32
    linalg.yield %5 : f32
  } -> tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %elemwise into %shared[%1, %2] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
此过程可能会导致操作数张量中的某些元素在循环的每次迭代中被（重新）计算。这也称为重新实现，它表达了执行冗余计算与将其结果存储在（慢速）内存之间的权衡。
```
## Linalg Ops 的简写“命名”形式 
```
Linalg 为常见情况（如矩阵乘法、点积、卷积等）提供了一组预定义运算。这些运算等同于上述generic运算，但无需详细说明访问模式和主体。例如，矩阵乘法很简单：

%matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                        outs(%init: tensor<8x10xf32xf32>) -> tensor<8x16xf32>
```
