## mojo的语法
- 基本上和python一致
- 可以使用rust的fn定义，此类函数则和rust一致，有内存借用管理逻辑
- def 表示 Python 模式, 弱类型
- fn 表示 Strict 模式, 强类型 + 内存安全
## mojo和rust的语法对比
- 所有权&T 借用的语法差异
```mojo
fn add(borrowed x: Int, borrowed y: Int) -> Int:
    return x + y
```
```rust
Rust 这里是&T
```
- 所有权&mut T,借用并修改
```mojo
fn add_inout(inout x: Int, inout y: Int) -> Int:
    x += 1
    y += 1
    return x + y
```
```rust
Rust 里这是 &mut T
```
- move语义
```mojo
fn set_fire(owned text: String) -> String:
    text += " "
    return text
```
```rust
```
- 未来可能生命周期管理会用
```mojo
struct Pair [type: AnyType, '2_first | '3_second]:
  first: inout'2 type
  second: inout'3 type
从 PLT 的角度来说, 按照现有语义, 要求这个类型系统的有 type, lifetime, mutable, complete 四种 kind, 所以不可能会比 Rust 简单, 还是得做好心理准备啊.

```
## mojo和python的差异
- __init__: Python 型构造，init 就是 UnsafeCell, 允许虽然没完全初始化, 但是让编译器认为已经完成初始化
- __copyinit__: 拷贝构造，copy 就是创建新的值不影响原来的.
- __moveinit__: 移动构造，move 就是 Rust 式 move, 当值生命周期结束时将值从一个所有者给到另一个所有者
- __takeinit__: 转义构造，take 就是 C++ 式 move, 值转移后, 析构过程还可以执行任意操作.
- 排列组合可以构建，不可实例化类型, 静态类型, 不可移动(pin)类型, 不可拷贝类型, 仅移动类型, 值类型, 以及窃取类型.
