<a name="BigTitle"></a>


English | [快速上手](#quickstart) | [数据](#data) | [赞助](#sponsorship) | [人员](#contributor) | [引用](#cite) 

# 骆驼QA: Better Conversational Question Answering Model with Answer Completion

骆驼QA是孙骜, 廖健生, 陈舒年, 李鲁鲁开发的，阅读理解并问答的语言模型。


<details>
  <summary> 每一个作者都是第一作者，顺序是随机的。(点这里具体)</summary>

李鲁鲁发起了项目，提出了拆分问题和使用Prompt Engineering生成更Diverse问题的方法。

孙骜完成了Prompt的设计，并增广了CoQA数据集，提出了基于问题转化的训练方案

廖健生编写了训练架构，并完成了模型的训练。

陈舒年（将要）完成了Q-A的Embedding初步实验，并编写了可视化

</details>

骆驼QA是Luotuo(骆驼)的子项目之一，后者由李鲁鲁，冷子昂，陈启源发起的

<p align="center">
    <img src="image/projBar.png">
</p>

骆驼QA是指


+ If you find this helpful, please star our major repo [Luotuo(骆驼)](https://github.com/LC1332/Luotuo-Chinese-LLM), Thanks Very Much

+ 如果你感到这个页面对你有帮助，拜托您去我们[骆驼的主页](https://github.com/LC1332/Luotuo-Chinese-LLM)也点上star，非常感谢！

---



## 发布TODO

- [ ] README页面装修
- [ ] inference脚本 colab
- [ ] Gradio脚本 colab
- [ ] (opt)更好的一个Gradio设计
- [ ] 更多截图





---

## Gradio 的需求

输入: 文本框

输入: 问题

输出: GLM6B的默认输出

输出: 一次问题转化之后的问题
输出: 问题转化之后的回答

输出: 将转化之后的问题 强行再编辑为原问题，然后再进行的回答

相当于我们的模型给两次回答。


---




# 骆驼QA : 仅使用6B语言模型的阅读理解系统

孙骜, 廖健生, 陈舒年, 李鲁鲁

每一个作者都是第一作者，顺序是随机的。

李鲁鲁发起了项目，提出了拆分问题和使用Prompt Engineering生成更Diverse问题的方法。

孙骜完成了Prompt的设计，并增广了CoQA数据集，提出了基于问题转化的训练方案

廖健生编写了训练架构，并完成了模型的训练。

陈舒年（将要）完成了Q-A的Embedding初步实验，并编写了可视化

## 引言

本文是对于我们骆驼QA模型，一个基于GLM-6B的中文阅读理解模型，的简短技术汇报。在语言模型的垂类应用中，关于一段特定文本的理解和QA是一个很重要的问题，有以下原因: 1. 大量涉密企业、或者是金融企业并不能够把自己的机密文档输入到公开的API中。2. 训练或者tuning一个语言模型的成本仍然很高，对于很多专业领域，如果能够在回答问题之前给定一段特定上下文的文本，可以使得模型有快速适应特定垂直领域的能力。3. 给定的文本的事实可以被用户控制，而不像语言模型的输出完全依赖模型的记忆和概率模型，可以让模型在对话中给出更符合事实的回答。

所以在驼铃的这个实验中，我们试图让一个6B的模型，通过Low Rank Adaptation的微调，争取实现对一段特定长度（大约200-600字）的阅读理解的能力。这对于我们之后完成更大的文本的搜索-QA系统，是一个重要的初步尝试。

注意到，完成一个中文的阅读理解模型是不琐碎的。1、过往的基于阅读理解的中文QA没有较大样本量的数据，很少有超过10万量级别的数据。2、类似CoQA的提问方式较为单一，有时候在连续对话中，会省略完整的问题，这往往与真实人类的对话习惯不一致。 3、相比于使用较为复杂的结构，我们希望去探究类似GPT这样的纯Decoder模型的性能。因为这种模型在一个真实的文档QA系统中，往往会有更好的表现。

由于CoQA数据集上的性能已经接近饱和，对于本个工作，我们主要希望研究三点 

1、区别于研究连续对话的机器人，我们将CoQA中的短问题延展为了更为准确的问题。并且在首先的测试中，我们对模型进行了单次提问的测试，我们希望消除连续提问对于性能的影响。我们希望测试相比于GPT这样的大模型，一个较小的模型是否能够完成阅读理解并针对文中的信息的提问进行回答。这对于后期制作更有专业性的模型更为重要。

2、对于CoQA的本来问题，我们进一步对每一个问题进行了5个增广，我们期望更多样的问题，能够带来模型对于数据集外的阅读理解有更好的性能。

3、对于连续对话的阅读理解，基于我们补全的问题，我们希望设计一个架构，模型能够先输出补全的问题，再对补全的问题进行回答。我们期望这种系统相比于直接回答短问题的系统能够表现出更好的性能。

所有的工作包括测试代码、训练代码以及数据将在清理后逐步开源。希望能够帮助中文大语言模型开源社区的发展。

## 相关工作

## CoQA数据集的增广

CoQA数据集经过简单清洗，共有7012个story，其中每个story中包含十余个问题。

增广工作包含两部分：

1. 对于短问题的补全：CoQA中包含着一些小于4个单词的超短问题，这些问题对于单个问题的问答来说包含的信息量过少。因此需要将这些问题补全为长问题。

2. diverse增广：对于每一个问题，我希望得到与其回答一致的另外的问法。这一种方式可以引导其对问题本身进行rethinking，作为中间任务的一个guiding

2023/04/10
目前英文数据集完成上述过程，正在进行中文翻译

## 训练

CoQA原来

```
起始prompt 文本 结束prompt

Q:问题1
A:回答1

Q:问题2
A:回答2

Q:问题3
A:回答3
```

单次回答

```
起始prompt 文本 结束prompt

Q:问题1
A:回答1
```

实验设计可以有两种：

1. 单阶段生成任务 
```
给你下面的文本和问题，请先给出一个对应问题的同义转述，再给出问题的答案。
文本为：{story}
原始问题为：{Q}
问题转义为：{CQ}（generate）
答案为{A}（generate）
```

2. 双阶段生成任务

第一次：
```
给你下面的文本和问题，请给出问题的同义转述
文本为：{story}
原始问题为：{Q}
同义转述为：{CQ}（generate）
```

第二次：
```
给你下面的文本和问题，请给出问题的答案
文本为：{story}
问题为：{Q}
这个问题也可以理解为：{CQ}
这个问题的答案是：{A}（generate）
```

### 连续问答的训练。

## 实验

### Metric

### 单次问答实验

### 连续问答实验