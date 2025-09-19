---
title: ai中的编程课程笔记（2025fall）
publishDate: 2025-09-14 
description: '爱编程'
tags:
  - ai
  - 底层  
heroImage: { src: './thumbnail.jpg', color: '#B4C6DA' }
language: '中文'
---
# 0.Introduction
目标：从cuda底层学习深度学习框架如何构建，并最终自己构建一个深度学习框架。  
# 1.Parallel Programming
## 1.1 why GPU&parallel programming
背景：在现代，越来越多的任务需要处理大规模的数据并且需要提高计算效率。为了解决这个问题，人们有两种思路：一种是提升计算频率，在一定时间内执行的计算次数增加；一种是讲任务分割进行并行编程，分别处理。  
对于第一种方法我们发现，由于能源消耗过大和难以散热的限制，计算的Clock Frequency（主频，计算效率）遇到了瓶颈，难以大幅增长  
相反，对于方法二，随着工艺的提升，计算单元的面积在逐渐缩小，芯片上面可以容纳的晶体管数量在增加，这些晶体管可以并行地处理任务，从而使得并行计算能力有充足发展空间。于是，我们需要重视并行计算这种加速模式。  
CPU和GPU虽然都具备并行计算的能力（CPU的多核），但是两者还是各有侧重：  
* CPU的电路更加复杂，在单任务的计算上灵活性和性能远超GPU，但是显然的，在能源消耗上也非常大，并行计算能力远弱于GPU  
* GPU有着更加丰富的并行计算电路，能效比（Power Efficiency）高，用更少的电量完成更多的工作，但是在处理问题上不如cpu灵活，并且GPU的编程模式更加受限，只在处理特定的任务时才有优势。  

* CPU追求延迟（latency）优化：The time required for each task
*  GPU追求吞吐量（throughout）提升：The total tasks per time unit  
## 1.2 Cuda programming
一个典型的Cuda程序分成两部分：CPU部分和GPU部分  
CPU被称为Host，GPU被称为Device。从名字上也可以看出，整体的运转主要由CPU统揽全局。  

CPU 的工作 是管理 GPU 内存和启动内核（Kernel） 。具体步骤包括：

* 在 GPU 上分配内存 。
* 将数据从 CPU 复制到 GPU 。
* 启动 GPU 内核 。
* 将结果从 GPU 复制回 CPU 。  

GPU 的工作 
* 是并行运行大量的内核 。
什么是内核？什么是线程(thread)？  
![Local Image](src/assets/imgs-aicode/1.png)  
如图：  
线程是GPU上的一个最小运算单元（图中的波浪线），多个线程构成了一个线程块（block），多个线程块构成一个网格（grid）。这是线程的组织形式  
而一个内核（Kernel）对于一个线程(thread)来说，就像一个 C 程序，每个线程都并行地执行这个kernel程序。  
例如：
```cuda
kernel_function<<< gridDim, blockDim >>>(args);
```
这就是在调用一个kernel函数，其中gridDim和blockDim分别表示网格和线程块的大小（gridDim表示网格中block的数量，blockdim表示block中线程数量），args表示传递给kernel函数的参数。让threads并行地执行kernel函数。  
而具体定义kernel中，我们使用threadIdx.x来表示当前线程在他的线程块中的索引（非全局索引），从而体现出每个thread执行相同函数同时又负责任务中的不同部分，进而达到并行效果。
```
__global__ void relu_gpu(float* in, float* out) {
 int i = threadIdx.x;
 out[i] = in[i] > 0 ? in[i] : 0;
 }
```
我们实现了一个并行的relu效果。  
tips：由于物理上的限制（芯片本身限制），所以blockDim一般是要固定的比如256/512，多了会超（这个由gpu本身决定）  
## 1.3 GPU Memory and Hardware
在GPU的储存中，tensor的储存方式在物理上表现为一块连续的内存，但是逻辑上，我们将其分成多个部分，每个部分储存一个tensor。  
![Local Image](src/assets/imgs-aicode/2.png)
右边的size(表示tensor的形状)，stride（表示在每一个维度上，tensor的一格变化对应连续的物理内存的变化），type（表示储存的数据的类型），从而我们可以直接用这几个量直接来表示tensor的所有信息。所以之后涉及的一些tensor操作，比如切片，旋转之类的，只要改这几个量就可以了，物理内存并不发生变化（可以理解为只修改了索引），于是相比于原来我们可能通过循环来进行的tensor修改，这种方法的时间复杂度为O(1)。（这就是为什么写pytorch的时候不要用循环，要用tonsor操作）  

### 1.3.1 3 kinds of memory of gpu
和cpu读取内存一样，GPU也有自己的显存（虽然技术语境下说的和日常使用并不太一样）  
主要有local shared global三种形式的memory  
* local ：最快，单个thread自己的memory
* shared ：较快，一个block中多个线程共享的
* global：最慢，全局共享的，容量也最大，一般说的显存大小指的是这个
Highlight: 我们可以通过将数据从全局内存转移到共享内存来进行加速  
### 1.3.2 Coalesced Global Memory Access
![Local Image](src/assets/imgs-aicode/3.png)
三种内存访问/写入形式：连续的，规律跳格，随机的，三种模式的效率依次递减  
所以，我们在设计并行程序的时候应当尽可能地使得内存的并行使用连续。  
例如图中的  
```cpp
t=x[i]
```
这里的每个i在并行的运行中相当于分别对应x[0],x[1],x[2]...(每个并行程序的索引不同，访问内存也不同)，于是是连续的，效率最高  
```cpp
t=x[i*2]
```
这个就是stride了  