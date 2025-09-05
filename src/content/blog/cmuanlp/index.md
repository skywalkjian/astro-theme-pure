---
title: Notes of CMU ANLP 
publishDate: 2025-07 
description: 'cmu高级自然语言处理'
tags:
  - ai
  - nlp  
heroImage: { src: './thumbnail.jpg', color: '#B4C6DA' }
language: '中文&ENG'
---

# Notes of CMU 11-711: Advanced Natural Language Processing (ANLP)（combined with 2024spring 2024fall 2025spring）



## 0. before you read
当下以大语言模型llm为主导的的nlp领域发展迅速，日新月异，笔者从课程安排来看，几乎每年都有变化，内容也会有所变动。所以这份笔记也许也应当具有时效性并随着时间推移进行迭代。但无论如何，姑且记录，作为个人学习梳理

## 1. Introduction to NLP
nlp(natural language processing) 自然语言处理，研究如何让计算机理解人类语言的学科
### 1.1 自然语言处理问题总的的来说可以归结为如下几个问题

1. 处理分析语言（给一段话，理解这段话的内容，指出其情感等，文本分类）
2. 人与机器交互（llm回答问题，生成代码）
3. 协助人人交互（语法检查，翻译）
4. 以及cv nlp结合 img和文本之间的转换

### 1.2 现在做nlp是要做什么

为什么有些模型在有些方面表现得好（追求sota：state of the art ）

为什么现在的sota模型在一些方面仍然有问题？

我们该如何改进我们的模型？

### 1.3 构建nlp系统的方法
1. 规则式构建模型（base on rules：如池袋，用一个固定的rule进行规范分类）不用经过训练
2. 对没有训练的模型进行提示（大白话就是提示词工程！通过设计提示词，让llm在不再次训练的情况下达到我们的要求）
3. 训练微调（微调：fine-tuning）
不只是llm，一些其他的基于训练的语言模型都是如此

### 1.4 动手构建最简单的规则式语言模型

如：一个判定情感的model

五步走：

1. 利用一个函数，提取特征（features）如：设定看到正向词语加一分，反之扣一分
2. 算得分
3. 构建决策函数，根据得分算出对应的结果
4. 准确性分析：根据计算出来的结果和实际数据结果对比，测评准确性
5. 根据准确性测评结果，进行误差分析并进一步修改

当我们重复这个循环之后，我们的模型在训练数据上面（train set）已经表现得准确度很高了，那么我们把它拿到测试数据上（test set）测评,再根据结果进行更改变化。是经典的 训练 测试 验证 逻辑。

然而规则式语言模型会有如下问题：

1. 低频词难以处理
2. 合成词难以处理（意思相近形态不同）
3. 否定词对句子产生的影响难以处理
4. 隐喻类比（整体句意无法处理）
5. 其他语言无法处理

因而引出：nlp based on Machine Learning

### 1.5 ML for NLP

#### 1.5.1 第一个尝试：词袋模型（basgs of words,BOW）

每个词对应一个独热向量（one-hot vector），把句子中所有词向量加起来就是代表句子特征的向量，
乘以权重W

$Wx=scores$

达到分类效果。通过ML，改进W矩阵

此处算法原理很朴素，每次训练结果对了词语的权重就加分反之减分，最后多轮训练可以通过最终输出的分数判断结果

#### 1.5.2 BOW的缺陷

1. one-hot向量编码的问题：无法处理近义词和词语变形，词语数量一旦大了，词向量的长度会非常大，非常低效且浪费内存空间  
2. 无法处理but，否定等句意相关信息，对于词语位置没有感知力

#### 1.5.3 改进：基于神经网络的模型

1. 通过*复杂方法* 把word编码成词向量
2. 通过*方法* 把词向量提取为句子特征
3. 根据神经网络处理句子特征

之后我们的研究其实都基于此，无论是transformer还是别的，本质上是对步骤一和二中*方法* 的改进，我们要找到一个可以提取语言特征的工具

## 2. Word Representation and Text Classifiers

### 2.1 Subword Models（子词模型）

#### 2.1.1 SM原理
为了改进one-hot向量，采用“字节对编码”（Byte Pair Encoding, BPE）  
基本思路是统计句子中的字母组合出现次数 如：es er 然后将持续最多的组合记为一个子词，然后将子词作为整体，再持续循环，最终得到可以用来拆分词语的子词表，内容比如 ：er est pro ed  
于是可以处理词语变形，同时节省内存（可以用少量子词表示大量词语）  

在获得子词表之后，我们以一元语言模型（Unigram LM，后面涉及）为例，来说明如何进行子词分割  
通过一元语言模型，我们进行一些算法（这不太重要），最终通过ml获得一个 最优词汇表 可以通过它以及一些计算得到每个子词出现的概率  
然后我们检查目标句子，进行不同的拆分方式，比如：est 拆成 e st 还是es t 最后选用概率最大的拆分方式就是结果

#### 2.1.2 注意事项 
多语言方面，容易过度分割混合语言语料中的小语种  
解决思路：对小语种进行采样
在如 es t和e st分割抉择上容易出问题  
解决思路：通过“子词正则化”，在训练时对不同的分割结果进行采样以减少鲁棒性

### 2.2 Continuous Word Embeddings（连续词嵌入）

对one—hot vector（独热向量）的大改进，使得用来表示各种词的向量长度大大减小，内存占用减少，同时具备了一些良好的性质，比如：“mom”-“female”和“dad”-“male”的词向量相似，近义词的词向量相似等  
我们会得到一个词向量库，每次解码（将词转化为向量）只需要查找（look up）到相应向量即可

### 2.3 如何训练更加复杂的模型（ML基础回顾）
不再赘述

### 2.4 Basic Idea of Neural Networks(for NLP Prediction Tasks) 
神经网络到底在干什么？

关键在于理解：提取并组合“特征” 

![Local Image](src/assets/images/1.png)

所有我们的基于深度学习的任务都可以归纳为利用各种神经网络架构去提取句子的特征（呈现为一个向量）（图中左侧部分）。这个向量包含了语言的features，也就是这句话的所有的信息，之后我们再通过神经网络提取出为了完成我们目标任务所需要的信息，并根据此即可完成任务（体现为得到scores，然后依据他来得到一个结果）.

每一层的神经网络可以视为提取一个层次的特征，从低阶到高阶，第一层可能只是局部特征，比如词组结构，之后层数增加就可以在原有基础上提取更加抽象的特征，比如句子结构等关系。

特征有高阶有低阶，例如，每一个连续词向量的每一个维度都代表着一种特征，多轮学习之后的隐藏层向量的每个维度可能蕴含着特征，但是这种所谓特征并不是人为规定，而是经过ml之后自动学习出来的

## 3. Language and Sequence Modeling

### 3.1 language models
分为生成式语言模型与判别式语言模型，本质都是概率语言模型  
$x$~$P(X)$ (X是词语或句子)  
生成式：预测下一个词  
判别式：预测label的概率进行分类  

### 3.2 Auto-regressive Language Models（自回归模型）

$$
P(X) = \prod_{i=1}^{I} P(x_i \mid x_1, \dots, x_{i-1})
$$

$x_i$ :next token
$x_i \mid x_1, \dots, x_{i-1}$:context

那么我们的关键在于，如何找到计算$P(x)$的方式

#### 3.2.1 Unigram Language Models（单元语言模型）

core：最简单的语言模型，基本思想是，每一个词都与其他词独立，不会互相干扰  
$P(x_i | x_1, \dots, x_{i-1}) \approx P(x_i)$
著名例子：信息检索tf-idf算法  

#### 3.2.2 小tricks：log参数化
$$
P(X) = \prod_{i=1}^{I} P(x_i \mid x_1, \dots, x_{i-1})
$$
序列一大就会导致P(x)极其小，超出计算机能力范围。  
于是通过取对数，把乘积问题转化为加法问题，之后处理梯度爆炸和梯度消失的时候也会运用对数参数化方法  
$P(X) = \prod_{i=1}^{|X|} P(x_i) \longrightarrow \log P(X) = \sum_{i=1}^{|X|} \log P(x_i)$  

#### 3.2.3 n-gram Models（n元语言模型）

在Unigram Language Models基础上加强了，认为 一个词出现的概率只与它前面固定数量的几个词有关
> **3.2.3.1 llm解释多元与单元语言模型的区别（参考资料）**
> 单元语言模型和多元语言模型是两种不同类型的语言模型，它们在结构和工作方式上有显著区别。
> 
>  **单元语言模型(Unigram Language Model)**
>
> 单元语言模型是一种最简单的语言模型。它假设句子中的每个词都是独立出现的，与其他词没有关联。换句话说，一个词出现的概率只取决于它自身在整个语料库中出现的频率。
>
> **特点:**
  **独立性假设:** 忽略词与词之间的顺序和上下文关系。
  **计算简单:** 计算一个句子的概率时，只需将每个词的概率相乘即可。
  **局限性:** 无法捕捉语言中的语义和语法结构，因为它不知道词语之间的联系。
>
> **举例:**
> 假设我们有一个句子：“**我 喜欢 吃 苹果**”。
在一个单元语言模型中，它的概率计算方式是：
$P(\text{我 喜欢 吃 苹果}) = P(\text{我}) \times P(\text{喜欢}) \times P(\text{吃}) \times P(\text{苹果})$
>
> 其中，$P(\text{我})$、$P(\text{喜欢})$ 等概率都是根据每个词在训练语料库中出现的频率来计算的。这种模型无法区分“我喜欢吃苹果”和“苹果喜欢吃我”这样的句子，因为它只关心每个词出现的概率，而不关心词语的排列顺序。
>
> ---
>
> ##### 多元语言模型 (N-gram Language Model)
>
> 多元语言模型是单元语言模型的扩展。它假设一个词出现的概率只与它前面固定数量的几个词有关，这个数量就是“n”。常见的有多元语法模型（N-gram），其中n=2时称为二元语法（Bigram），n=3时称为三元语法（Trigram），以此类推。
>
> **特点:**
> * **上下文依赖:** 考虑词语的顺序和上下文，能捕捉更丰富的语义信息。
> * **计算复杂:** 随着n的增加，需要计算和存储的概率数量会呈指数级增长。
> * **更接近自然语言:** 相比单元模型，它能更好地反映人类语言的语法和句法结构。
>
> **举例:**
依然是句子：“**我 喜欢 吃 苹果**”。
在一个**二元语法（2-gram**模型中，它的概率计算方式是：
$P(\text{我 喜欢 吃 苹果}) = P(\text{我}) \times P(\text{喜欢}|\text{我}) \times P(\text{吃}|\text{喜欢}) \times P(\text{苹果}|\text{吃})$
>
> 这里，$P(\text{喜欢}|\text{我})$ 表示在“我”这个词出现之后，“喜欢”这个词出现的条件概率。这个模型考虑了前一个词对当前词的影响，所以它能更好地理解“我喜欢吃苹果”这个句子，因为它知道“我”后面出现“喜欢”的概率要比出现“苹果”的概率高得多。
>
> 随着n的增加，模型的表现会越来越好，但也面临“数据稀疏”问题，即很多n-gram组合在训练数据中可能从未出现过，导致概率为零。现代的大型语言模型（LLM）则超越了传统的N-gram模型，使用了更复杂的神经网络架构（如Transformer），能够处理更长的上下文和更复杂的语言依赖关系。

#### 3.2.4 多元语言模型的问题
1. 近义词还是处理不了    
2. 间隔词处理不好 (intervening words)：指的是在两个有强关联的词之间，存在其他不相关的词语。比如Mr.Perter Smith and Mr. Jane Smith,中间的词没有关系，但是产生较大影响  
3. n元规模不可能无限扩张，一般到7差不多了，所以对于长上下文依赖很难处理

#### 3.2.5 优势
那为什么我们还会用到他呢？  
因为他相比神经网络类语言模型更加高效快速，在处理一些比较简单的语言任务中发挥更好的性能，对计算资源要求低。所以我们经常会用他来处理原始的上游数据，经过n元语言模型处理后的数据在用神经网络语言模型

#### 3.2.6 基于神经网络的模型

初期：特征化模型，为后来的高级模型提供铺垫。例子：前馈神经网络语言模型

![Local Image](src/assets/images/2.png)

将词向量拼接成一个长向量，然后通过tanh激活函数和W1将其转化为低阶的隐藏层，在这个过程中W1的每一行都与行向量进行点积并加以偏置，获得了两个词向量的融合特征（原本的词向量，每个维度都代表着这个词的某一种特征），得到的新向量每一行都是新的融合特征，随后再来一次获得scores。这样，相似的词语有着相似的词向量，隐藏层也相似。进而可以处理同义词。而通过向量拼接并一起进行机器学习，可以将连续的几个词同时作为输入，也可以解决干预词的问题。但是由于拼接无可能无限长，而且必须认为预先定好数量，所以还是无法解决长距离语义依赖的问题

#### 3.2.7 进阶：真正意义上成熟的序列模型

1. RNN（循环神经网络）&LSTM（Long short time memory）  
2. CNN（卷积神经网络）  
3. attention

#### 3.2.7.1 RNN: 

![Local Image](src/assets/images/4.png)

模型之类的不再赘述，关于反向传播倒有些新意：三个中间参数W，在各个时间步中保持一致，一次更新全部更新  

双向RNN：单项rnn智能让模型利用前面的信息，双向rnn让模型可以兼顾前面的信息和后面的语义信息，因为前后文对理解一个句子之中的词义都有帮助（有点像掩码模型的思路）

RNN的缺陷：梯度爆炸。  
推导  
for the k step: 
$$h_k=\tanh(W_x x_k + W_h h_{k-1} + b)$$
$$O_k=W_x x_k+W_h h_{k-1} + b$$
$$\frac{\mathrm{d}L_k}{\mathrm{d}W_x}=\frac{\mathrm{dL_k}}{ \mathrm{d}h_k}\cdot\frac{\mathrm{d}h_k}{\mathrm{d}W_x}$$ 

$$\frac{\mathrm{d}h_k}{\mathrm{d}W_x}=\frac{\mathrm{d}h_k}{\mathrm{d}O_k}\cdot\frac{\mathrm{d}O_k}{\mathrm{d}W_x}=\frac{\mathrm{d}h_k}{\mathrm{d}O_k}\cdot (x_k+\frac{\mathrm{d}W_hh_{k-1}}{\mathrm{d}W_x})=\frac{\mathrm{d}h_k}{\mathrm{d}O_k}\cdot (x_k+\frac{\mathrm{d}h_{k-1}}{\mathrm{d}W_x}W_h)$$
$$\frac{\mathrm{d}h_k}{\mathrm{d}O_k}  \ is\  equal\  to\  \frac{\mathrm{d}\tanh\theta}{\mathrm{d}\theta}$$
$$ Recursion:\frac{\mathrm{d}h_k}{\mathrm{d}W_k}=\frac{\mathrm{d}h_k}{\mathrm{d}O_k}\cdot (x_k+\frac{\mathrm{d}h_{k-1}}{\mathrm{d}O_{k-1}}\cdot (x_{k-1}+\frac{\mathrm{d}h_{k-2}}{\mathrm{d}W_x}W_h)W_h)$$
$$
\frac{\mathrm{d}h_k}{\mathrm{d}W_x} = \frac{\mathrm{d}h_k}{\mathrm{d}O_k} \cdot \left( x_k + \frac{\mathrm{d}h_{k-1}}{\mathrm{d}O_{k-1}} \cdot \left( x_{k-1} + \frac{\mathrm{d}h_{k-2}}{\mathrm{d}O_{k-2}} \cdot \left( x_{k-2} + \cdots \right) W_h \right) W_h \right)
$$

递归展开后，梯度项会不断乘以 $W_h$，最终公式为：
$$
\frac{\mathrm{d}h_k}{\mathrm{d}W_x} = \sum_{t=1}^{k} \left( \prod_{j=t+1}^{k} W_h \cdot \frac{\mathrm{d}h_j}{\mathrm{d}O_j} \right) \cdot \frac{\mathrm{d}h_t}{\mathrm{d}O_t} \cdot x_t\\ (in\ fact\ x_t\ is\ x_t^T)
$$

当 $|W_h| > 1$ 时，$\prod_{j=t+1}^{k} W_h$ 随 $k$ 增大呈指数级增长（梯度爆炸）；当 $|W_h| < 1$ 时，$\prod_{j=t+1}^{k} W_h$ 随 $k$ 增大呈指数级减小（梯度消失）。

#### 3.2.7.2 LSTM:

在RNN的基础上，提出了LSTM进行改进

![Local Image](src/assets/images/3.png)


$X_t$is the input vector  
$h_t$is the hidden state like RNN  
$C_t$is the cell state,which can store information for long periods of time(like memory)

有三个“门”：遗忘门，输入门，输出门

遗忘门：$C_t-1$进来之后，和$f_t=\sigma(W_f\dot{[h_{t-1},x_t]}+b_f)$相乘

值得指出的是，$\sigma$  (sigmoid)函数的性质导致了向量上的数值在0~1，于是这个$f_t$向量中的每一个元素都代表了一个遗忘因子：如果某个元素的值接近 1，这意味着对应的旧信息（细胞状态中的一个维度）应该被完全保留，或者说，几乎不被遗忘。相反，如果某个元素的值接近 0，这意味着对应的旧信息应该被完全遗忘，或者说，被丢弃。

遗忘门其实是LSTM和RNN最大区别之一，通过这种遗忘的机制，有效保留了重要的信息忽略了长期记忆中需要丢弃的信息，从而使得长期记忆成为可能

输入门：也就是把现在的输入$x_t$和之前的隐藏状态$h_{t-1}$结合起来，经过一个sigmoid函数，得到一个0~1之间的向量$i_t$，这个向量决定了当前输入的信息中哪些部分是重要的，应该被写入到细胞状态中。然后通过一个tanh函数，将$x_t$和$h_{t-1}$结合起来，得到一个新的候选值$\tilde{C}_t$，这个候选值包含了当前输入的信息。最后，将$i_t$和$\tilde{C}_t$相乘，得到当前输入中应该被写入细胞状态的信息。
然后将这个信息加到用遗忘门更新后的细胞状态中,更新完毕：
$$C_t=f_t*C_{t-1}+i_t*\tilde{C}_t$$

输出门：取舍新的细胞状态（记忆）中的信息，我们得到了输出$h_t=\sigma(W_o\dot[h_{t-1},b_o])*\mathrm{tanh}(C_t)$

LSTM的优势在于缓解了RNN的梯度消失问题，使得模型能够更好地捕捉长期依赖关系。
如何实现？我们看到：  
$$\frac{\mathrm{d}L}{\mathrm{d}C_t}=\frac{\mathrm{d}L}{\mathrm{d}h_t}\cdot\frac{\mathrm{d}h_t}{\mathrm{d}C_t},it\ can\ be\ solved\ easily$$  
进行梯度回传  
$$\frac{\mathrm{d}L}{\mathrm{d}C_{t-1}}=\frac{\mathrm{d}L}{\mathrm{d}C_t}\cdot\frac{\mathrm{d}C_t}{\mathrm{d}C_{t-1}},we\ focus\ on\ \frac{\mathrm{d}C_t}{\mathrm{d}C_{t-1}}$$

$$
\frac{\mathrm{d}C_t}{\mathrm{d}C_{t-1}} = \frac{\mathrm{d}(f_t * C_{t-1} + i_t * \tilde{C}_t)}{\mathrm{d}C_{t-1}} = f_t + C_{t-1} * \frac{\mathrm{d}f_t}{\mathrm{d}C_{t-1}} + i_t * \frac{\mathrm{d}\tilde{C}_t}{\mathrm{d}C_{t-1}} + \tilde{C}_t * \frac{\mathrm{d}i_t}{\mathrm{d}C_{t-1}}
$$

我们来看看这些复杂项的链式法则展开：

* $$\frac{\partial f_t}{\partial C_{t-1}} = \frac{\partial f_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial C_{t-1}}$$
* $$\frac{\partial i_t}{\partial C_{t-1}} = \frac{\partial i_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial C_{t-1}}$$
* $$\frac{\partial \tilde{C}_t}{\partial C_{t-1}} = \frac{\partial \tilde{C}_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial C_{t-1}}$$

---

这些项都涉及到了隐藏状态对细胞状态的导数 $\frac{\partial h_{t-1}}{\partial C_{t-1}}$。而 $\frac{\partial h_{t-1}}{\partial C_{t-1}} = \frac{\partial}{\partial C_{t-1}}(o_{t-1} * \tanh(C_{t-1}))$，这个导数包含输出门 $o_{t-1}$ 和 tanh 的导数 $1 - \tanh^2(C_{t-1})$。这两个项的元素值都小于1。因此，这些间接路径上的梯度在回传过程中会不断乘以小于1的值，导致其指数级地衰减。随着时间步的增加，这些间接路径的梯度贡献会迅速变得微不足道,于是：

$$\frac{\mathrm{d}L}{\mathrm{d}C_{t-1}}\approx f_t$$

由于我们处理的上下文肯定是有逻辑的，（如果上下文非常无关的话，遗忘门中的元素会频繁出现趋近于0，但是这种文本数据价值本身就很低）所以遗忘门中的元素大多趋近于1（大多数的的信息还是会被保留下来），于是防止了梯度爆炸 （$\sigma$函数结果在0~1之间），同时减缓了梯度消失

#### 3.2.7.3 CNN:  

CNN在语言处理中很简单，就是简单的设定滑动窗口大小然后逐一采集特征：  
$$h_t=conv(W_x[x_{t-k};x_{t-k+1};...;x_{t+k}])$$

如果是自回归模型进行语言生成的话，就改成：
$$h_t=conv(W_x[x_{t-k};x_{t-k+1};...;x_{t}])，只取前面的$$

CNN的用途：可以提取出几个连着的token的特征，这样可以提取出词组的信息，可以构建出文字组合的联系，但是还是无法解决长距离依赖的问题，而且各个词的位置信息还是没能体现

#### 3.2.8 模型的效率评估和提升tricks
指标：  
参数量（Parameter count）： 指的是模型中可训练的参数总数。参数量越少，模型通常越小，需要的内存和计算资源也越少，但模型的表达能力可能受限。  
内存使用（Memory usage，主要看峰值）： 指的是模型运行时占用的内存大小，包括模型本身以及在推理（或训练）过程中所需的临时内存。  
延迟（生成第一个token，到最后一个token所花的时间）  
吞吐量（Throughput）： 指的是在单位时间内模型能处理的请求（或数据）总量。吞吐量越高，说明模型在处理批量任务时越高效，能够更好地服务多个并发用户。  
蒸馏/压缩（distillation/compression）”和“生成算法（generation algorithms）都是为了提高模型的效率和性能而提出的技术手段。  
#### 3.2.8.1 mini-batching
由于现代计算硬件在处理矩阵问题方面GPU比CPU更好，例如对于10个数据，一次性处理10个数据要比10次处理每个数据要快，因此可以通过将多个样本组合成一个小批量（mini-batch）来提高训练效率。小批量训练可以充分利用GPU的并行计算能力，从而加速模型的训练过程。在训练过程中常用地方法是进行concatentate（进行矩阵拼接）
#### 3.2.8.2 其他优化tricks
1.不要把同样操作放入循环内，不如提前算好  
2.循环可以用拼接代替  
3.减少CPU和GPU之间的数据移动，最好一次性完成  

## 4. Attention and Transformer Models
### 4.1 **attention**
### 4.1.1 attention基本思想
query（查询向量）,key（键向量）  
语言序列中地每个token对应一个key向量，通过将q和k建立关联，得到注意力分数，
注意力分数通过softmax函数进行归一化处理，得到每个token在当前上下文中的重要性权重。  

>**例如**：**假设你正在使用一个模型来生成一张图片的文字描述**（**Image** >**Captioning**）。
>
>**输入**：
>
>序列A (查询 Query)： 待生成的文字描述（例如，一个不完整的句子，比如“一只>猫...”）
>
>序列B (键 Key 和值 Value)： 一张照片的视觉特征向量（例如，通过卷积神经网络从图片中提取的一系列特征，每个特征向量代表图片中的一个区域，比如猫的耳朵、眼睛、背景等等）。  
>其中K Q V的计算均通过$K=W_kx$,$Q=W_qx$,$V=W_vx$进行机器学习得到
>
>**工作流程**：
>当模型想要生成下一个单词时，它会进行以下操作：
>
>查询（Query）:
>模型会使用当前已生成的文字信息（例如，“一只猫”的向量表示）来生成一个查询向量。
>
>键（Key）与值（Value）:
>同时，图片中的每一个视觉特征向量（比如猫的眼睛、嘴巴、尾巴等）都扮演着键和值的角色。
>
>计算注意力权重:
>模型将查询向量（代表“一只猫”）与图片中所有的键向量（代表眼睛、嘴巴、尾巴等）进行比较。
>它会发现，“一只猫”这个查询与猫的眼睛、胡须和尾巴这些区域的键向量最相似。因此，这些区域会得到最高的注意力分数。
>
>加权求和，生成上下文向量:
>这些注意力分数被转换为权重后，模型用这些权重对所有视觉特征向量的值（v）进行加权求和。最终，模型会得到一个上下文向量。
>这个上下文向量集中了图片中最相关的视觉信息，也就是猫的眼睛、胡须和尾巴的特征。
>
>生成下一个词:
>模型将这个上下文向量与已生成的文字信息结合起来，预测下一个最有可能的单词是“在”或“躺在”。

### 4.1.2 self-attention and cross-attention
上述例子就是一个基本的cross—attention的例子，一个语言序列中的词向量作为q，另一个序列中的词向量作为k，对于第一个序列中的tokens，我们找到在第二序列中各tokens对应的注意力分数，也就是最相关的信息点。常用于处理不同信息转换，经常用于机器翻译，以及多模态转换（比如图像信息转为文字信息）。  
而self-attention，就是始终以同一个序列作为对象，kq均从同一个序列中产生，于是我们最后得到的注意力分数描述的是对每一个token他在本身在一个句子中的地位以及相关的其他信息。现代大语言模型比如gpt系列都在用自注意力机制  
将所有的这些每一个token得到的注意力分数进行汇总，就可以得到整个句子的信息，也就是上下文向量。

### 4.1.3 Attention Score Functions
我们前面只说了kq之间可以建立关系得到注意力分数，但是具体的计算方法有多种。  
* **Multi**-**layer** **Perceptron**(**MLP**)
$$f(q, k) = W_1\text{tanh}(W_2[q,k])$$
    Flexible, often very good with large data（flexible运用了非线性的tanh使得学习能力更强）
* **Bilinear**(双线性函数)
$$f(q, k) = q^T W k$$
    More efficient than MLP, but less flexible（）
*  **Dot Product**
$$f(q,k)=q^Tk$$
    No parameters! But requires sizes to be the same.
* **Scaled Dot Product**
$$f(q,k)=\frac{q^Tk}{\sqrt{|k|}}$$
    Problem: scale of dot product increases as dimensions get 
    larger 
    Fix: scale by size of the vector
### 4.1.4 Masking for Training
这是一种对训练数据的处理方法  
![Local Image](src/assets/images/5.png)  
在我们给模型投喂训练数据的时候，如果投喂整个句子的话，那么前后文都都会被看到，那就达不到我们想要的，希望模型能够通过预测未来词语来进行学习进步的效果（相当于抄答案了）。但是我们也希望：在训练模型（比如 Transformer）时，我们能通过一次性进行大规模的矩阵乘法来提高效率，而不是一个词一个词地循环处理。  
于是如图中所展示，我们把后面的词通过掩码（masking）把他们遮住（对应方法应该是用极小复数填充），然后将每个掩码的句子合并为一个大矩阵进行训练

### 4.2 transformer
![Local Image](src/assets/images/6.png)
理解transformer的关键就是理解这幅图。  
首先我们看到两种类型的transformer：Encoder-Decoder Model （左）和 Decoder-Only Model（右）。  
区别在于，左边的模型同时使用了编码器和解码器，先将输入语句进行向量化，再将再将其编码变成包含了句意信息的上下文向量，然后加入中间的多头注意力进行交叉注意力机制处理。于是这个上下文向量就变成了后续文本生成的极大依赖（由他产生k，v），q由下面的output向量产生。最开始给的是一个特殊的起始符向量（shifted right：将输入序列向右移动一个位置，并在序列的开头添加一个特殊的起始符，比如 <sos> (Start of Sequence) 或 <bos> (Beginning of Sequence)，以便最开始的生成），然后不断生成延长。  
由于这种方式对于编码器处理的input向量极度依赖，，所以常用于机器翻译等需要严格按照输入规则来的任务。  
右边是纯解码器模型（Decoder-Only Model），它只使用了解码器部分。输入是一个已经存在的文本序列，目标是生成下一个最有可能的词。解码器通过自注意力机制来关注输入序列中的所有词，并生成下一个词的概率分布。由于不需要编码器的上下文信息，这种模型在处理生成任务时更加灵活。同时由于注意力机制只依赖于本身生成的内容，更加适合处理文本生成等规则更少更灵活的任务。  

>我们就用现在的大语言模型来举例子，同样是输入prompt。Encoder-Decoder Model会将prompt进行编码，得到一个上下文向量，然后依赖上下文向量通过交叉注意力机制来生成后续文本。而Decoder-Only Model则直接使用prompt作为输入，将其视为之后生成内容的一部分，通过自注意力机制来生成下一个token。
### 4.2.1 how transformer work(details)
>    **Core** **Transformer** **Concepts:**
>* Positional encodings(位置编码)
>* Multi-headed attention（多头注意力）
>*    Masked attention（掩码注意力）
>*    Residual + layer normalization（残差连接&层归一化）
>*    Feed-forward layer（前馈层）

两种transformer本质一样，我们以全解码模型为例展开分析。  
1. 首先进行word embedding,也就是之前讲过的词向量化操作，可以运用之前的知识（2.2节），不再赘述  

2. 进行位置编码（positional encoding）。(解释清楚位置编码需涉及到多头注意力机制的原理，可以先往后看) 由于注意力机制对每个词向量进行kqv的查询，我们会发现对于一个同样的词，他在前后文中的表现出的语义完全相同，我们无法区别他们，但是实际上，两者的意思肯定会有差别。  
比如说两个“big”在不同上下文中的含义可能完全不同，一个可能是修饰“big cat”，另一个可能是“修饰big data”，这就需要位置编码来帮助模型区分。  
所以我们会在词向量化的时候在原有矩阵$W$的基础上加一个包含位置信息的矩阵$W_{pos}$，最终得到的词向量为$(W + W_{pos})x$。    
    >位置编码的方式有两种，一种是学习式的位置编码，就是将位置编码矩阵$W_{pos}$作为一个可训练的参数矩阵，和词向量矩阵一起进行机器学习。另一种是固定式的位置编码，就是通过一些数学函数来生成位置编码矩阵，比如说正余弦函数：  
    >$$PE(pos,2i)=\sin(pos/10000^{2i/d_{model}})$$  
    >$$PE(pos,2i+1)=\cos(pos/10000^{2i/d_{model}})$$  
    >其中$pos$是词在句子中的位置，$i$是词向量的维度索引，$d_{model}$是词向量的总维度。这样，每个位置都会有一个独特的编码，这个编码会随着位置的变化而变化，从而帮助模型区分不同位置的词。  
    >这种正余弦函数的位置编码方式有一个好处，就是它可以让模型更好地捕捉到词之间的相对位置关系，因为正余弦函数具有周期性，可以表示不同位置之间的距离关系。
3. 进行多头注意力机制处理：首先介绍普通的transformer的单头注意力机制。  
   首先，和之前所述的attention机制一样（此处我们的注意力计算函数设定为点乘方式  
   $$scores = {Q \cdot K^T}$$  
   然后，为了防止点积过大导致的梯度爆炸，我们对scores进行缩放处理，并且进行softmax归一化,得到注意力权重：  
   $$attention\ weights = \mathrm{softmax}(\frac{scores}{\sqrt{d_k}})$$  
   最后，加权求和，我们引入v，将注意力权重乘以各个词向量的值，得到每个词的上下文向量：  
   $$Output = \mathrm{softmax}(\frac{scores}{\sqrt{d_k}}) \cdot V$$  

   而多头注意力机制本质上是对矩阵的一种“拆分”，比如，将一个长为1024的向量拆分为4个长为256的矩阵。  
   >举例：  
   原来对于KQV，我们得到（为了简化假设KQV都一样）：  
   >$$ K，Q，V=\begin{pmatrix}
    1&2&3&4\\ 5&6&7&8    
   \end{pmatrix}$$  
   >这里的矩阵是2*4的矩阵，表示有2个词，每个词向量长度为4。  
   那么我们将其拆分为4个2*1的矩阵：  
   >$$
    K_1，Q_1，V_1=\begin{pmatrix}
    1\\ 5
    \end{pmatrix},K_2，Q_2，V_2=\begin{pmatrix}
    2\\ 6
    \end{pmatrix},K_3，Q_3，V_3=\begin{pmatrix}
    3\\ 7
    \end{pmatrix},K_4，Q_4，V_4=\begin{pmatrix}
    4\\ 8
    \end{pmatrix}
   >$$  
   >然后分别计算4个注意力分数,和前面所示的单头注意力机制一样，最后再把得到的4个输出矩阵按照词向量维度方向拼接起来，得到多头拼接的输出矩阵，最后再通过一个线性层进行最终的整合，得到最终的输出。  

   **为什么要用多头注意力机制？**      
   这4个矩阵原来因为softmax会被归一化，互相产生牵绊因而发挥相近的作用，对于语义理解来说，只是参数量增多，可以承载的信息更加丰富，但是本质上还是处理同种信息。但是，拆分之后，我们对每个矩阵的kqv进行不同的线性变换（也就是乘以不同的权重矩阵），同时归一化是互不干扰，这样就可以让每个矩阵学习到不同类型的信息，从而提升模型整体对于语义的理解能力。我们认为：每个“头”都独立地执行一次单头注意力，关注输入序列中的不同子空间信息。例如，在一个句子中，一个头可能专注于捕捉语法关系（主谓宾），而另一个头可能专注于捕捉语义关系（同义词、反义词）。  





