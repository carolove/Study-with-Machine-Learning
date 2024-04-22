# mlir tutorial 学习纪要 6 学习mlir 的Traits系统
- 搞清楚什么是mlir的traits系统
- @llvm-project//mlir:SideEffectInterfacesTdFiles 这个里面的SideEffectInterfaces怎么解释和翻译
- def PolyOrContainer : TypeOrContainer<Polynomial, "poly-or-container">; 怎么理解poly-or-container，特别是其中container这个概念
- [Pure, ElementwiseMappable, SameOperandsAndResultElementType]，这个语法表达，以及其中的单词概念完全搞不懂
- 其次也没有搞懂新增的mlir测试案例的作用

## 什么是traits和interfaces
- 首先，traits和interfaces都是mlir提供的，用于代码复用的解决方案
- 对于interfaces，Inteface分为Dialect和Op两类，其中后者的粒度更细。
```
比如Ch4里演示的内置的DialectInlinerinterface和ShapeInferenceOpInterface，而且也有基于TableGen的设计方法！

struct ToyInlinerInterface : public DialectInlinerInterface { 这是tblgen 语法中继承 DialectInlinerInterface的方式
  using DialectInlinerInterface::DialectInlinerInterface; 具体的话就是声明使用 DialectInlinerInterface ，让ToyInlinerInterface具有DialectInlinerInterface的多有复用逻辑
  //...
};
```
- you can implement interfaces on operations and types
-  a trait is an interface with no methods. Traits can just be “slapped on” an operation and passes can magically start working with them. They can also serve as mixins for common op verification routines, type inference, and more.
-  怎么理解呢，就是说traits他不需要实现，只需要贴在对应的types、operations上，就可以正常使用
-  \[Pure\] traits 是 NoMemoryEffect 内存无关性？ 和 AlwaysSpeculatable 可预测性的. 这样这个pure trait是可以轻易的加到operation上的
