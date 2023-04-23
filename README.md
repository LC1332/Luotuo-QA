<a name="BigTitle"></a>


English | [快速上手](#quickstart) | [数据](#data) | [赞助](#sponsorship) | [人员](#contributor) | [引用](#cite) 

# 骆驼QA: Better Conversational Question Answering Model with Answer Completion

骆驼QA是孙骜, 廖健生, 黄泓森 , 陈舒年, 李鲁鲁开发的，阅读理解并问答的语言模型。


<details>
  <summary> 每一个作者都是第一作者，顺序是随机的。(点这里具体)</summary>

李鲁鲁发起了项目，提出了拆分问题和使用Prompt Engineering生成更Diverse问题的方法。

孙骜完成了Prompt的设计，并增广了CoQA数据集，提出了基于问题转化的训练方案

廖健生编写了训练架构，并完成了模型的训练。

黄泓森翻译了增广后的CoQA数据。

陈舒年（将要）完成了Q-A的Embedding初步实验，并编写了可视化

</details>

骆驼QA是[Luotuo(骆驼)](https://github.com/LC1332/Luotuo-Chinese-LLM)的子项目之一，后者由李鲁鲁，冷子昂，陈启源发起。

<p align="center">
    <img src="image/projBar.png">
</p>

骆驼QA是指给定一段特定的文本，用户针对文本中的内容，进行一个提问。语言模型试图理解文本中的内容，对用户的问题进行回答。这里我们从陈丹琦学姐参与的CoQA数据集出发，基于唐杰老师实验室发布的GLM6B模型，建立了中文的骆驼QA模型。


+ If you find this helpful, please star our major repo [Luotuo(骆驼)](https://github.com/LC1332/Luotuo-Chinese-LLM), Thanks Very Much

+ 如果你感到这个页面对你有帮助，拜托您去我们[骆驼的主页](https://github.com/LC1332/Luotuo-Chinese-LLM)也点上star，非常感谢！

<a name="quickstart"></a>

## 快速上手

|  | Colab链接 | 细节 |
| --- | --- | :--- |
| 测试脚本 | - | 当前的骆驼QA为0.1模型 |
| 交互界面 | - | 基于0.1模型，用Gradio开发的交互界面 |

## 例子输出

<p align="center">
  <img src="https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/image/QAResult.png">
</p>


## 训练

我们利用Alpaca-LoRA和GLM6B+LoRA（驼铃）的方案，在增广后的CoQA英文和中文数据集上，各训练了一个英文和中文的模型。


<a name="data"></a>

## 数据

例子数据

```
Text: 曾经有一只住在森林里一个小洞穴里的小熊。他的洞穴舒适、温暖、黑暗，洞前还有一点院子。小熊和他的父母一起生活，白天到处走动，晚上卷缩着睡觉。他喜欢寻找浆果吃。他最喜欢的浆果是蓝莓，但无论他找到什么浆果，像草莓、覆盆子、樱桃等，都会吃。 \n\n小熊的洞穴附近有一条河，他喜欢坐在河岸边看鱼和青蛙，看自己在水中的倒影。在一个阳光明媚的下午，当他凝视河流时，他看到了一群鸭子在游泳。他起身跟随它们。它们在河里游泳，他在岸边走动。直到他们到达森林里的一个小空地为止，它们一直这样旅行。小熊停下脚步，环顾四周，看到空地里完全长满了蓝莓，比他以往见过的还要多！ \n\n小熊吃饱了蓝莓，然后带回了尽可能多的蓝莓。他快乐地上床睡觉了。这是一个美好的一天。",

Question 1-A: 这只熊居住在什么类型的住所？
Question 1-B: 这只熊居住在什么类型的住所？
Question 1-C: 这只熊居住在什么类型的住所？
Answer: 在一个洞穴里。

Question 2-A: 熊在他的洞穴里的伙伴是谁？
Question 2-B: 熊洞里的其他居民是谁？
Question 2-C: 这只年轻的熊和谁分享了它温暖舒适的洞穴？
Answer: 他的母亲和父亲
```

CoQA数据集经过简单清洗，共有7012个story，其中每个story中包含5个左右的问题，每个问题进行了5次增广。

我们准备公开这批数据，正在研究CoQA和OpenAI的数据协议，并且准备申请共享这批数据的表格和网站，之后就会释放这批数据。


<a name="sponsorship"></a>

## 赞助(Sponsorship) 骆驼项目

在训练骆驼嵌入的时候，完全使用社区捐赠的经费购买的远程服务器

所有的CoQA的数据增广和翻译都使用了社区捐赠的经费购买了OpenAI的API。

如果你有兴趣赞助骆驼项目，请点击[主项目](https://github.com/LC1332/Luotuo-Chinese-LLM#%E8%B5%9E%E5%8A%A9sponsorships)或者查看[赞助表单](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/Sponsorship_and_balance.md)

If you are interested in sponsoring the [Luotuo Project](https://github.com/LC1332/Luotuo-Chinese-LLM#%E8%B5%9E%E5%8A%A9sponsorships), please click on the [major project](https://github.com/LC1332/Luotuo-Chinese-LLM) or view the [sponsorship form](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/Sponsorship_and_balance.md).

[回到开头](#BigTitle)

<a name="contributor"></a>

## 人员

更详细的整个骆驼项目相关的人员信息，请查看[骆驼项目的主页](https://github.com/LC1332/Luotuo-Chinese-LLM#%E8%B4%A1%E7%8C%AE%E8%80%85contributors)

每个作者都是第一作者，顺序是随机的。

李鲁鲁发起了项目，提出了拆分问题和使用Prompt Engineering生成更Diverse问题的方法。

孙骜完成了Prompt的设计，并增广了CoQA数据集，提出了基于问题转化的训练方案

廖健生编写了训练架构，并完成了模型的训练。

黄泓森翻译了增广后的CoQA数据。

陈舒年（将要）完成了Q-A的Embedding初步实验，并编写了可视化

[回到开头](#BigTitle)


<a name="cite"></a>

## 引用

如果您在项目中使用了我们的模型、代码或者数据，请引用下面第一篇文章。

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca,
  author={Ao Sun, Jianshen Liao, Hongsen Huang, Shunian Chen, Cheng Li},
  title = {Luotuo-QA: Better Conversational Question Answering Model with Answer Completion},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LC1332/Luotuo-QA}},
}
```

```
@misc{alpaca,
  author={Ziang Leng, Qiyuan Chen and Cheng Li},
  title = {Luotuo: An Instruction-following Chinese Language model, LoRA tuning on LLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LC1332/Luotuo-Chinese-LLM}},
}
```

[回到开头](#BigTitle)

---



## TODO

- [x] README页面装修
- [ ] inference脚本 colab
- [ ] Gradio脚本 colab
- [ ] (opt)更好的一个Gradio设计
- [ ] 更多截图
- [ ] 在colab完成之后，做一些有趣的例子Good & Bad

以上是4/25之前

---

以下是之后

- [ ] 进一步清理翻译的CoQA数据，引入翻译API进行核对补充，引入中英文Embedding进行进一步比对
- [ ] 训练得到0.3模型
- [ ] 寻找一些没有QA的语料（1万文本以上，最好3个domain），让GPT生成QA，加入到训练数据中
- [ ] 训练得到0.7模型
- [ ] 进行初步的搜索+QA的尝试




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