# ACLIB-GNN

ACLIB-GNN 将图注意力与对抗学习结合，用于可解释的节点分类。模型通过两层图注意力网络编码节点，以 k-hop 邻域构建候选解释子图，并对池化后的子图嵌入施加扰动，联合优化节点与子图预测。

模型包含三个部分：

* **图编码：** 利用多头图注意力学习节点表示。
* **子图聚合：** 对以目标节点为中心的 k-hop 邻域进行池化。
* **对抗学习：** 使用训练标签生成子图嵌入扰动，联合优化节点和子图目标。

[English](README.md)

## 安装

所有命令均在仓库根目录运行。创建环境并安装依赖：

```bash
conda create -n aclib-gnn python=3.9 -y
conda activate aclib-gnn
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

使用 CUDA 12.1 时，将 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

## 实验设置

默认训练 100 轮，隐藏维度为 64，第一层使用 4 个注意力头，邻域为一跳，每个邻域最多保留 10 个节点。

实验采用传导式五折分层交叉验证，默认随机种子为 42。每折从外层训练节点中划分 10% 作为独立验证集，用于选择最佳模型。测试标签不参与训练和扰动生成，支持 CPU 与 CUDA 运行。

## 数据准备

支持 **Cora、Citeseer、Pubmed 和 Coauthor CS**。首次训练时由 PyTorch Geometric 自动下载，需要联网。

```text
data/
├── Cora/
├── Citeseer/
├── Pubmed/
└── CS/
```

可通过 `--data-dir` 指定其他数据位置，保留 PyG 自动生成的内部目录结构。

## 预测评估

先完成模型训练，再加载保存的模型，在对应测试集划分上评估。源码包不包含预训练权重。

```bash
python evaluate.py --run-dir results/Cora/paper --seed 42 --fold 1
python evaluate.py --run-dir results/Citeseer/paper --seed 42 --fold 1
python evaluate.py --run-dir results/Pubmed/paper --seed 42 --fold 1
python evaluate.py --run-dir results/CS/paper --seed 42 --fold 1
```

评估输出 Accuracy、宏平均 Precision、Recall、F1、one-vs-rest AUC 和 AUPR。训练过程也会自动评估并保存各测试折的结果。

## 模型训练

```bash
python train.py --dataset Cora
python train.py --dataset Citeseer
python train.py --dataset Pubmed
python train.py --dataset CS
```

模型权重、数据划分、预测和指标保存到 `results/<dataset>/<objective>/`。每个随机种子和折次分别保存，`all_metrics.csv` 记录各折结果，`summary.json` 汇总均值和样本标准差。

运行单折实验或多个随机种子：

```bash
python train.py --dataset Cora --fold 1 --output-dir results/cora-fold1
python train.py --dataset Cora --seeds 42 43 44 --output-dir results/cora-multiseed
```

每次运行需使用空输出目录。全部参数见 `python train.py --help`。

## 子图导出

训练时指定目标节点，导出候选邻域：

```bash
python train.py --dataset Cora --fold 1 --export-nodes 0 10 42 --output-dir results/cora-subgraphs
```

对应折次目录中的 `subgraphs.json` 包含节点编号、诱导边及预测标签。这些子图是候选支持结构，不是经过验证的因果解释。当前版本未实现解释排序 AUC 和保真度评估。

## 代码结构

| 路径 | 功能 |
| --- | --- |
| `train.py` | 模型训练与交叉验证 |
| `evaluate.py` | 已保存模型的评估 |
| `aclib/model.py` | 图编码器、分类器、扰动与损失 |
| `aclib/data.py` | 数据读取、邻域构建与池化 |
| `aclib/metrics.py` | 分类指标 |
| `scripts/` | 数据集快捷入口与打包工具 |
| `.github/workflows/` | 自动软件检查 |

不下载数据集也可检查代码和训练流程：

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --output-dir results/smoke-check
```

## 复现状态

默认 `--objective paper` 对应正文公式（3）–（8）；`--objective script` 保留原始脚本的损失公式，并修正训练与评估问题。目前尚未验证论文中的数值结果，未包含图分类和基线对比实验。

详细差异见 [复现核对说明](docs/REPRODUCIBILITY.md)，软件检查结果见 [验证记录](docs/VALIDATION.md)。开源许可证待作者确定。

## 引用

使用本实现时，请引用论文：

YunXia Li, Chuya Wu, Mengjie Cai, Quan Lu and Zhenyu Yue. *ACLIB-GNN: Incorporating Adversarial Causal Learning into Information Bottlenecks for Interpretable Graph Neural Networks.*

## 联系

* **Zhenyu Yue：** [zhenyuyue@ahau.edu.cn](mailto:zhenyuyue@ahau.edu.cn)
