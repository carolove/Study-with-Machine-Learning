# trition学习（一）

## 环境准备
- 硬件环境 n卡8G 40系列
- ubuntu 22.04
- gcc 11.4.0
- clang 14.0.0
- python 3.10.12
- cuda 12.1
## 源码下载 
- llvm-project 使用gh下载，项目很大，使用git下载容易中断 (gh repo clone llvm/llvm-project )
- triton  使用gh下载，同上 (gh repo clone openai/triton)
## 版本选择
- triton学习选择 2.2.x 分支
## 编译过程
### 第一步 构建llm 
- 打开triton项目选择合适的llvm版本
