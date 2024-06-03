# 对dialect op的理解
- 在dialect定义op的过程中，可以用td文件来快速定义和描述该dialect下的op的行为操作
- 也可以通过手工编码的方式来定义

 ## 定义的dialect op可以完成两方面的工作
 - dialect定义的op，可以完成从从mlir 源代码文件的读取和mlir汇编代码解析，从而将mlir文件，转化为llvm/mlir程序运行中的对应的dialect 的op的运行时c++对象，从而方便在c++程序中对对象进行操作，包括重写pattern rewrite和conversion等
 - 也可以完成c++运行时 dialect op对象，对mlir文件的文本输出/导出
 - 这样也就明确了dialect本身的边界，就是完成对mlir文件到c++运行时对象的相互映射，方便进行c++运行时的操作
 - 如果要完成其他类型的操作比如重写pattern rewrite和conversion，应该放在conversion目录下来完成，可以将运行时对象做重写和替换
