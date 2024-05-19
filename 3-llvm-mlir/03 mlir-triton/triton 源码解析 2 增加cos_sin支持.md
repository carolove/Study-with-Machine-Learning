# triton 源码解析之增加cos-sin语义支持

## 版本信息
- 主分支v2.1.0
- commit: 2824345065a5e26464c1ba4e62cf04f07b50ff3a
- 名称 [LANGUAGE] Added cos/sin (#132)

## 源码学习目标
- 学习在triton中增加一个算子，所需要的源码修改
- 源码修改的结构
- 在源码主要层次/模块（frontend/core/backend/driver）等修改逻辑

## cos/sin算子修改的模块
- frontend/python binding 语义支持
- core/ir GPU dialect 语义支持以及lowering支持

## cos/sin frontend前端支持
### python binding
- python/triton/language.py 加入两个python buildin函数调用前端定义的函数
- python/src/triton.cc frondend 加入cos sin两个函数的定义，然后会调用core/ir提供的dispatch逻辑

## cos/sin core/ir模块逻辑
- lib/ir/dispatch.cc   主要作用就是在core/ir层定一个入口
- lib/ir/builder.cc  会把从python调用  builder 指令层
- lib/ir/instructions.cc 会通过 指令层 真正将python dsl开始调用生成层
- lib/codegen/selection/generator.cc 这是将python dsl通过生成，生成到llvm ir，后继通过gpu dialect生成最终的ptx码，整个generator层才是真正实现dsl解析
  
