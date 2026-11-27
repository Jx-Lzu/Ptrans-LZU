PTrans 训练数据补充包
整理日期 2026年11月27日

用途
补齐已有 PTrans GitHub 项目 research/FILES.sha256.json 中列出的8个数据文件。
这是既有项目的补充包，不重复包含模型权重、完整训练代码或原始实验CSV。
8个文件逐字节沿用现有研究数据，已与原研究校验清单的大小和SHA-256核对一致。

使用方法
1. 将本ZIP解压到已有仓库根目录，也就是包含 predict.py、research/ 的目录。
2. 解压后，8个数据文件应位于 research/data/；两个准备脚本位于 research/。
3. 在仓库根目录运行：
   python verify_training_data_supplement.py
   该检查只需Python标准库，不需要安装PyTorch或启用GPU。
4. 后续训练继续使用原项目的环境及研究入口。检查数据本身不会启动训练。

本包8个原始数据文件
  ptrans_normalized_dataset.jsonl.gz  12223348 bytes
  ptrans_split_manifest.csv  5601729 bytes
  ptrans_smiles_aug_train_v1.jsonl.gz  1033373 bytes
  ptrans_smiles_aug_train_v1.meta.json  1001 bytes
  train_molecular_library.npz  1631031 bytes
  train_molecular_library.meta.json  1418781 bytes
  ptrans_spectral_anomalies.csv  9435 bytes
  ptrans_summary.json  616 bytes
8个数据文件合计 21919314 bytes。

附带的原有准备脚本
research/ptrans_prepare.py
  从已有五列原始CSV读取三核峰表，进行规范化、质量标记和数据划分。
  它不负责从上游单核谱记录重建三核配对关系。
research/ptrans_prepare_smiles_aug.py
  基于固定训练划分生成等价SMILES缓存，需要同目录原项目已有的
  ptrans_smiles_aug.py 和 ptrans_train_baseline.py。
  当前训练分子特征库以原缓存原件提供；这两个脚本不声称一键重建该分子库。

可选的准备命令
已有数据可直接使用，无需重新生成。若仅需检查准备过程，应输出到新目录：
  python research/ptrans_prepare.py --input data/raw/experimental.csv.gz --outdir prepared
  python research/ptrans_prepare_smiles_aug.py --project-root research --output-data prepared/ptrans_smiles_aug_train_v1.jsonl.gz --output-meta prepared/ptrans_smiles_aug_train_v1.meta.json
第一个命令要求将原始CSV先放在 research/data/raw/experimental.csv.gz。
准备脚本需要原研究环境的NumPy、pandas和RDKit等依赖；复现实验版本为
Python 3.10.8、NumPy 2.2.6、pandas 2.2.3、RDKit 2025.09.4。
重新生成文件的字节和哈希不保证与固定历史缓存相同，请勿覆盖本包原件。

校验内容
TRAINING_DATA_SUPPLEMENT.sha256.json：本包文件大小、SHA-256及来源。
verify_training_data_supplement.py：验证文件哈希、22,226条数据及8:1:1划分、
训练增强缓存ID、分子库训练范围、gzip完整性及NPZ内部CRC。
本包没有另行训练、评估或改变任何模型权重。
