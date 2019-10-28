

# 大数据中的文本挖掘·作业1

## Introduction

本次作业的任务是故事生成，即给定一个故事标题，要求输出5个句子的短故事。我们力图复现一下论文(Seq2Seq，静态两步式生成)，并尝试加了一些模块(Self-Attention)，以期在本任务上达到较好效果。

## Related Work

本次作业本质上是文本生成的任务，序列到序列[1]模型是文本生成领域比较常用的一个算法，也是课程提供的参考文献[2]中使用的方法，采取了编码器-解码器的架构，用编码器编码输入，解码器则用来产生输出，两个模块之间采用注意力模块相连。

参考文献[2]中提出了两步式的故事生成方法，即先对给定的标题生成一个简短的storyline，再通过storyline生成具体的故事。文中提出了两种故事生成的模型，即静态生成模型和动态生成模型。前者先使用序列到序列模型根据标题生成完整的storyline，再用storyline生成故事；后者则是交替式地动态生成storyline和故事。在此基础上我们做了一些调研。

 Ammanabrolu等人采用了一个级联的模型来完成给定故事开头续写故事的任务[3]。他们使用了Martin等人提出的event抽象结构[4]来表示句子，并将其进一步扩展。他们将故事生成的任务分成了生成event和生成故事两个步骤，与文献[2]采用中间结构storyline的思路相似。Yang等人提出了根据若干个主题生成文章的方法[5]。他们在decode生成文本的时候引入了外部知识，并且借用了seqGAN的训练方法增强模型表现。这些工作和本次作业一样，需要根据比较短的输入生成较长的文本。

本次作业采用的评测指标bleu值，全称bilingual evaluation understudy，由Papineni等人于2002年提出[6]，是一种常用于机器翻译等领域的自动评价指标，现也多用于各种文本生成任务的评价。

对于测试集中的每组数据，模型对于输入序列产生一个输出序列，这个输入序列对应一个或多个标准输出（因为机器翻译的任务并不是一对一的，一个句子可以有多种翻译方式，所以可以有多个标准输出）。其基本原则是希望机器翻译得到的译文与人工译文重合度尽可能高。具体评测时，会比较机器译文和参考译文之间的n-gram的重合度，即机器翻译中的n-gram在参考译文中的最大命中次数。n一般取1、2、3、4。但是这样会倾向于给较短的序列更高的分数，因此引入了长度惩罚因数BP。若机器译文长度小于参考译文，则令BP<1，会导致最终bleu评分降低。其余情况BP=1。最终计算公式可以表示为：

$$
bleu = BP\dot{}exp(\sum_{n=1}^N{w_nlog(p_n)})
$$
其中w表示各个n-gram的权重，一般都取为1/N，p表示各n-gram的命中率。N一般取为4，即bleu值最多只看机器译文和参考译文4-gram的重合程度。BP可以用以下公式表示：
$$
BP = 
\begin{cases}
1& {c > r}\\
e^{1-r/c}& {c \leq r}
\end{cases}
$$
其中，c表示机器译文的长度，r表示匹配程度最高的参考译文的长度。

因为n-gram的命中率p可能为0，导致对0取对数，因此在实际中会使用光滑函数[^7]进行特殊处理，保证对数中的自变量大于0。

bleu评分综合权衡了序列间的n-gram重合度和长度等因素，是一个被广泛使用的指标。但是它的一个比较明显的缺点是只会机械地比较模型输出和标准输出之间的n-gram重合度，无法正确比较两者在语义、情感等方面的相似性。不过这也是几乎所有自动评测指标共有的缺点。

## Data Analysis

本次作业采用ROCstories数据，共有98161组数据，其中前90000组用于训练，后8161组数据用于测试。我们对训练集中数据进行了分析。

### 2.1. Title Analysis

训练集中，标题组成的词表共有19349个不同的词，总共由196614个词语组成，标题的平均长度为2.18。其中出现频率前5的单词如下表所示：

| 单词     | the   | a     | new   | to    | day   |
| :------- | ----- | ----- | ----- | ----- | ----- |
| 出现次数 | 16638 | 4214  | 2884  | 1711  | 1261  |
| 出现频率 | 8.46% | 2.14% | 1.47% | 0.87% | 0.64% |

可见，出现次数前5的单词的总出现频率超过了10%。另一方面，有9087个单词在训练集的标题中只出现了一次，有13397个单词在训练集的标题中出现次数不超过三次，占了词表的69.24%。

统计了训练集中标题长度的分布，最短的标题长度为1，最长的标题长度为25。具体分布如下图所示：

![title-len](img/title-len.png)

可以看到，长度为2的标题数量最多，其次是长度为1,3,4,5的。长度小于等于5的标题占了所有标题的99.23%。

### 2.2. Story Analysis

故事组成的词表共有65336个不同的单词。总共由3936562个词语组成。在长度上，不同位置的句子的长度分布有比较明显的差别，各个位置句子的平均长度如下表所示：

| 句子位置 | 1    | 2    | 3    | 4    | 5    |
| -------- | ---- | ---- | ---- | ---- | ---- |
| 平均长度 | 7.90 | 8.82 | 8.89 | 8.80 | 9.33 |

所有句子的平均长度是8.75。可以看到，第一个句子的平均长度明显小于其他位置的。另一方面，各个位置句子组成的词典出现频率前10的词中有6个共有的，前20的词中有13个共有的，前100的词中有51个共有的。说明各个位置的句子在词表的分布上也有一定区别。

所有句子组成的词表中出现次数前5的单词如下表所示：

| 单词     | the    | to     | a      | was    | he     |
| -------- | ------ | ------ | ------ | ------ | ------ |
| 出现次数 | 196912 | 155817 | 127907 | 107254 | 103488 |
| 出现频率 | 5.00%  | 3.96%  | 3.25%  | 2.72%  | 2.63%  |

与标题的词表中一样，出现频率前5的单词占了总的词数中的相当一部分，超过了15%。另一方面，故事的词表中有24688个词只出现了一次，出现次数不超过三次的词有39042个，占了词表的59.76%。

训练集中句子的长度最短为1，最长为19，具体分布如下图所示：

![sent-len](img/sent-len.png)

句子长度的分布大致上比较接近正态分布，越靠近平均长度的数量越多。

## Work of This Repo

本来是想在作者的源码上改一下，加点其他模块。但是看了一下作者的源码，发现作者的源码里有相当多的操作非常迷幻，比如加了很多论文中没有提到的trick(如Weight Drop)，似乎没有在model中用Attention(但论文中明确提到有)等等。另一方面，从代码风格上来讲，作者的源码对我们而言确实不够平易近人，难以下手修改。总之，咱也看不懂，咱也不敢问，TAT

于是我们只能把baseline重写一遍。。。

### Description

按照论文中的描述，所谓Static式生成其实就是两步走，先Seq2Seq生成StoryLine，再Seq2Seq生成Story。本repo也只实现了基本的Seq2Seq(带Attention的)。

在实现过程中，参考了一些网上的风格较好的tutorial，基于pytorch和torchtext等封装较好的库。

### Usage

- 准备数据(`data`目录下)
  - 利用RAKE算法抽取关键词不在本repo范围内
  - 原始数据文件名为 /train/valid/test_title_line_story.txt
  - 可运行data_split.py把数据集按照不同的域切分, 保存格式为tsv
    - 如在title to story-line 过程中，只需使用`title`与`story_line`两个域，生成的样例可参见`train_title_line.tsv`

- 在config.py里设置参数，如数据集目录，模型超参数, 训练好的模型保存路径，生成的结果路径等
- 传入上一步config的名字，运行main.py, 可参考 `scripts_example` 中的`/title2line.sh`和`line2story.sh`
  - 可通过`mode`参数选择是否要进行训练或生成

### Some Tricks

- Vocabulary截断。本Repo并没有使用预训练好的词向量，而本任务又难以从头开始学到足够好的词向量，尤其是某些低频词。所以我们在构建词典的时候把很多低频词都扔掉了。(如果不扔的话，训练结果会非常差)
- 抑制重复。在生成故事线的过程中，生成的5个词往往会重复，论文中在decode的时候暴力去重，我们也沿用了这样的方法。

### Advanced Attmpt

考虑到只重写了一个baseline，没有一点儿花里胡哨的东西，面子上总是挂不住滴。由于时间有限，我们也没有尝试加其他更复杂的东西，考虑到model architecture最容易解耦，所以尝试了一个较为花哨的model。

一言以蔽之，我们想把self-attention加到RNN里。self-attention最早被大家熟悉应该是在transformer系列工作中。transformer系列工作完全抛弃了RNN的循环思路，而是采用MutiHeadAttention模块作为基石。我们使用**多层RNN**作为Encoder，并在RNN层与RNN层之间插入了MutiHeadAttention模块。

<img src="img/self_attn.png" alt="self_attn" width="300" height="400" />

### Experiment

- 复现Baseline

  - 泛化性能差。具体表现在模型稍微一训练就会过拟合。trainset上的loss可以一直下降，但validset上的loss不到5个epoch就到达了最低点并开始上升。这意味着模型的容量很大，足以拟合训练集，但这种拟合是大大牺牲了泛化性能的。

  - 两步的loss如下图所示

    <img src="img/t2l.png" alt="283147ccb2d9cfc26e60fb93b815e1a" width="300" height="400" />

  - <img src="img\l2s.png" alt="74cef8d14c44ce809c64deef8b33556"  width="300" height="400" />

  - 可能是由于我们并没有像论文中一样添加了很多trick，所以我们复现的模型的bleu score的结果要比源码差一些

    | Model      | Bleu Score           |
    | ---------- | -------------------- |
    | 作者源码   | 0.011146367604214563 |
    | 我们的代码 | 0.006871091977711924 |

- 加入Self-Attention模块

  - 我们惊奇地发现加入此模块后，训练过程中valid loss不再飘
  - <img src="img/l2s_attn.png" alt="099077dff4c43b5ee6852782d2528b6"  width="300" height="400" />
  - 但很遗憾，我们发现decode的结果其实很差，每个story-line几乎都decode出了同样的结果。所以也解释了valid loss不飘吧。
  - 究其原因，应该是这个task本身的src很短，在encoder模块加很复杂的可学习模块其实更容易导致整个框架过拟合

## Conclusion

- 本task属于NLP中较为新型的一种task。source长度远小于target长度，此类任务我们认为一是难在评测，二是难在生成。
  - 评测上，论文中也提到用Bleu评测不好，难以反映出生成的实用效果。
  - 生成上，我们直觉上觉得基于这个数据集，基于简单的Seq2Seq很难做到很好的效果。基于无监督预训练的大生成模型(如GPT-2)才是未来。(但我们考虑到GPT-2生成的虽然通顺，但是bleu评测不见得高，所以就没跑GPT-2。。)
- 把代码重写一遍也算一个小小的contribution吧
  - 这份代码代码风格还算凑活，个人感觉比作者的源码更适合教学使用 (捂脸逃)
  - 安利一个很好的[tutorial](https://github.com/bentrevett/pytorch-seq2seq), 本repo的代码就是照着他的写的

## TODO

- 如果有同学发现有bug可以及时反馈，我们应该会维护此repo一两个星期吧



## Reference

[1]: Sequence to Sequence Learning with Neural Networks, Ilya Sutskever et al., 2014



[2]: Plan-And-Write: Towards Better Automatic Storytelling, Yao et al., 2019



[3]: Story Realization: Expanding Plot Events into Sentences, Prithviraj Ammanabrolu et al., 2019



[4]: Event Representations for Automated Story Generation with Deep Neural Nets, Lara J. Martin et al., 2018



[5]: Enhancing Topic-to-Essay Generation with External Commonsense Knowledge, Pengcheng Yang et al., 2019



[6]: BLEU: a method for automatic evaluation of machine translation, Kishore Papineni et al., 2002



[7]: A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU , Boxing Chen et al., 2014



