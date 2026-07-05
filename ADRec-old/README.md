# ADRec: 在序列推荐中释放扩散模型的威力 [![Static Badge](https://img.shields.io/badge/Cite--us-007ec6?style=flat-square&logo=google-scholar&logoColor=white)](#-citation) [ArXiv](https://arxiv.org/abs/2505.19544)

KDD 2025 论文《Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach》的官方实现。

作者：Jialei Chen, Yuanbo Xu✉ 和 Yiheng Jiang

<img src="README.assets/overview.svg" alt="overview" style="zoom:150%;" />

## 环境要求

必须安装以下环境包以设置所需的依赖项。

```
auto_mix_prep==0.2.0
einops==0.8.0
matplotlib==3.10.0
numpy==2.2.2
PyYAML==6.0.2
scipy==1.15.1
seaborn==0.13.2
torch==2.4.0
torchtune==0.4.0
tqdm==4.66.5
```

我们的代码已在配备 NVIDIA GeForce RTX 4090 GPU 的 Linux 服务器上测试运行。

## 使用方法

#### **首先，请导航到 `src` 目录。**

**我们提供了预训练的嵌入权重，可以直接用于后续的骨干网络预热和全参数微调。您可以直接运行以下命令进行模型训练和评估。**

#### ADRec:

```
python main.py --dataset baby --model adrec
```

#### 预训练嵌入：

如果您想复现预训练权重，可以运行以下代码：

```
python main.py --dataset baby --model pretrain
```

#### 使用多任务框架 PCGrad 的 ADRec：

```
python main.py --dataset baby --model adrec --pcgrad true
```



### 我们还发布了一些基线模型。

#### DiffuRec:

```
python main.py --dataset baby --model diffurec
```

#### DreamRec:

```
python main.py --dataset baby --model dreamrec
```

#### SASRec+:

```
python main.py --dataset baby --model sasrec
```



### **我们还提供了一个脚本，可以在多个数据集上运行多个模型。**

```
bash baseline.bash
```

#### 

### t-SNE 可视化

t-SNE 可视化实验可以通过 `/src/t-SNE.ipynb` 进行。

### 原始嵌入空间的综合评估

可以使用 `/src/embedding_metrics.ipynb` 对原始嵌入空间中的嵌入表示进行综合评估。

## 致谢

感谢 [RecBole](https://recbole.io/)、[DiffuRec](https://github.com/WHUIR/DiffuRec)、[DreamRec](https://github.com/YangZhengyi98/DreamRec) 和 [SASRec+](https://github.com/antklen/sasrec-bert4rec-recsys23)。

## 📄 引用

如果您觉得这项工作有用，请考虑引用我们的论文：

```
@inproceedings{JLchen2025ADRec,
	title={Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach},
	author={Jialei Chen and Yuanbo Xu and Yiheng Jiang},
	booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
	year={2025},
	organization={ACM},
	doi = {10.1145/3711896.3737172}
}
```
