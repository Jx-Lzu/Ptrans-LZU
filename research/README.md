# 研究复现

本目录保存六组完整系统的生成器训练、负样本挖掘、评分头训练、完整验证及冻结测试代码。日常预测使用仓库根目录的 predict.py。这里的训练程序需要有权使用的原协议研究文件。

原实验在 Linux 和 NVIDIA GPU 上运行，使用 Python 3.10.8、PyTorch 2.6.0+cu124、NumPy 2.2.6 和 RDKit 2025.09.4。训练环境另含 pandas 2.2.3。完整队列训练 HCP、HC、HP、CP、C 和 H 六组；本仓库提供 HCP 的推理权重。

若具备原实验数据及获准使用的缓存文件，把以下文件放入本目录下的 data/。

- ptrans_normalized_dataset.jsonl.gz、ptrans_split_manifest.csv
- ptrans_smiles_aug_train_v1.jsonl.gz、ptrans_smiles_aug_train_v1.meta.json
- train_molecular_library.npz、train_molecular_library.meta.json
- ptrans_spectral_anomalies.csv、ptrans_summary.json

FILES.sha256.json 中保留对应文件校验值。仓库根目录的 evaluation/split_ids.csv 只用于核对 ID 与分区，没有完整数据清单中的结构和谱图字段。若改用自有数据，请重新准备规范化记录、数据划分、训练词表、增强数据和分子库，并更新相应配置和文件校验值。

在仓库根目录启动：

~~~bash
python research/retrain_queue.py start --parallel 2
python research/retrain_queue.py status
~~~

每个生成器分三阶段训练 100、10、6 轮，共 128,992 次更新，按完整验证集最低损失选取检查点。排名系数与质量容差固定。各组评分头从该组生成器提出的候选中学习。六组完成验证之后锁定测试计划。根目录 evaluation/ 中的指标来自既有实验，本次发布打包没有另行训练。
