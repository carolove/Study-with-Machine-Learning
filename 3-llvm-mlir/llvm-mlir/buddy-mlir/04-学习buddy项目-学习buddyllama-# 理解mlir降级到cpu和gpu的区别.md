# 理解mlir降级到cpu和gpu的区别
- 在第三篇中我们已经学习了DynamoCompiler，理解了通过torch 的compile jit技术，可以将model 转化为 fx graph/aten ir，然后进一步convert到自定义mlir ir表示层（主要是linag dialect以及tosa dialect）上
