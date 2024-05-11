# 理解mlir降级到cpu和gpu的区别
- 在第三篇中我们已经学习了DynamoCompiler，理解了通过torch 的compile jit技术，可以将model 转化为 fx graph/aten ir，然后进一步convert到自定义mlir ir表示层（主要是linag dialect以及tosa dialect）上
- 这一章主要学习在获得mlir ir的情况下，如何生成target（比如cpu、gpu）设备运行码的过程和逻辑
- 其次，还需要通过不同的pr，主要是[cpu pr]()与[gpu pr]()，通过对两个pr的学习理解，弄清楚lowering到异构设备的流程和逻辑，以及主要的程序设计、算法设计
