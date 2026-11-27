# 研究复现

本目录保存六组完整系统的生成器训练、负样本挖掘、评分头训练、完整验证及冻结测试代码。日常预测使用仓库根目录的 predict.py。本完整包已随附原协议使用的八个固定研究数据文件；训练程序按保存的配置读取这些文件。

原实验在 Linux 和 NVIDIA GPU 上运行，使用 Python 3.10.8、PyTorch 2.6.0+cu124、NumPy 2.2.6 和 RDKit 2025.09.4。训练环境另含 pandas 2.2.3。完整队列训练 HCP、HC、HP、CP、C 和 H 六组；本仓库提供 HCP 的推理权重。

以下文件已经放在本目录的 data/，无需再次查找或复制：

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

## 研究依赖安装

在仓库根目录运行 `python -m pip install -r requirements-research.txt`。该清单复用基础依赖，并补充规范化脚本需要的 `pandas==2.2.3`。GPU训练仍需与原配置相同的PyTorch CUDA版本；文件与数据校验不需要GPU。

## 新增的准备与校验入口

仓库根目录运行 `python verify_training_data_supplement.py` 可检查八份数据、划分、训练缓存与文件完整性，不需要GPU。`ptrans_prepare.py` 是原三核五列表格的规范化脚本，`ptrans_prepare_smiles_aug.py` 是原训练等价SMILES缓存准备脚本。前者不负责追溯上游单核记录的三核配对；分子特征库按原缓存提供，这两个脚本不单独重建该分子库。

无需为了使用已提供数据而重新运行准备脚本。可选命令将输出写入新的 `prepared/`，详见根目录 `TRAINING_DATA_SUPPLEMENT_README.txt`。若重建数据或缓存，需重新核对配置及校验值，不要直接覆盖原件。
