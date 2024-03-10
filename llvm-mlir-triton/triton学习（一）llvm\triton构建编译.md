# triton学习（一）llvm\triton构建编译.md

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
- triton学习选择 2.2.x 分支 ，本次构建的triton的commitid：0e7b97bd47fc4beb21ae960a516cd9a7ae9bc060
## 编译过程，使用本地编译的llvm构建triton
- 整个编译过程不进行llvm等文件下载，国内环境比较恶劣，下载过慢，经常出错，因此采用如下过程
### 第一步 准备python venv环境
- 查看triton项目的readme找到《Install from source》章节，按照 《Or with a virtualenv:》构建一个venv环境，环境名默认为triton，并启用triton环境
```
git clone https://github.com/openai/triton.git;
cd triton;
git checkout release/2.2.x 
python -m venv .venv --prompt triton;
source .venv/bin/activate;

pip install ninja cmake wheel; # build-time dependencies
```
### 第二步 构建llvm
- 打开triton项目选择合适的llvm，查看triton的readme， 2.2.x分支下，查看 《Building with a custom LLVM》 可以获知对应的llvm版本信息，eg：5e5a22ca
- 按照章节说明，编译llvm
- 在$HOME/.triton/llvm下面建立一个对应的llvm软连接，eg: ln -s ${HOME}/{your_llvm_work_folder}/llvm-project/build llvm-5e5a22ca-ubuntu-x64
### 第三步 下载对应的pybind11
- 查看triton项目的python/setup.py文件，找到对应的pybind11信息，下载对应的版本并放到${HOME}/.triton/pybind11目录下
- 下载解压缩后，可以看到 eg：${HOME}/.triton/pybind11/pybind11-2.11.1
### 第四步 修改setup.py文件
- 删除代码中关于llvm pybind的下载相关代码，以及三个cuda生态圈工具的下载
- 修改记录见 https://github.com/carolove/Study-with-Machine-Learning/blob/main/llvm-mlir-triton/setup.py
### 安装triton到venv环境
- pip install -e python
### 验证
- 安装对应的python package， 可能会有torch、PySide2、pandas、matplotlib等
- 运行 python3 python/tutorials/03-matrix-multiplication.py
