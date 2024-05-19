# mlir tutorial 学习纪要 8 学习mlir 的 预检查器 td声明式与编码式
- td声明式
- 手动编码模式

## 自定义traits，通过附加traits的方式完成td声明式预检查模式
- 原理和编码
```
分为两部，第一部分，为实现自定义的traits
1、实现一个Has32BitArguments 继承了OpTrait，有一个verifyTrait函数，所能验证的范围应该是ins 和ous
2、在对应的op上附加该traits
```

## 手工编码式预检查模式
- 原理和编码
```
1、在对应的op ODS对象上要加入，let hasVerifier = 1
2、在对应的ODS 实现中要实现对应的接口，这个案例中就是实现 ::verify接口
```

## 学习两个方式互换
- td声明是预检查 Operands
- 手工编码式 预检查 point
