# triton 源码解析之增加cos-sin语义支持

## 版本信息
- 主分支v2.1.0
- commit: 2824345065a5e26464c1ba4e62cf04f07b50ff3a
- 名称 [LANGUAGE] Added cos/sin (#132)

## 源码学习目标
- 学习在triton中增加一个算子，所需要的源码修改
- 源码修改的结构
- 在源码主要层次/模块（frontend/core/backend/driver）等修改逻辑

## cos/sin算子修改的源码结构
