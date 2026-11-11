# PTrans · 三核 NMR 有机磷结构生成与精确质量重排

PTrans generates candidate structures from experimental ¹H, ¹³C and ³¹P peak lists and reranks them using a supplied neutral monoisotopic mass.

本仓库发布论文最终的 **HCP 完整系统**。三个生成器和两个评分头共含 13,831,313 个参数。模型根据三核峰表生成、评分、合并候选，再应用固定 ±5 ppm 的精确质量优先排序。运行预测只需要本仓库中的文件。

## 快速预测

推荐 Python 3.10。在仓库根目录运行：

~~~bash
python -m venv .venv
~~~

Linux 用 `source .venv/bin/activate` 激活，Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活。以下 PyTorch 安装命令适用于 Linux 或 Windows，按设备二选一。

CPU：

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
~~~

NVIDIA GPU，CUDA 12.4 版：

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~~~

安装其余依赖、校验文件并预测：

~~~bash
python -m pip install -r requirements.txt
python verify_release.py
python predict.py --input examples/input.json --output results/prediction.json --device auto --top-k 10
~~~

`--device` 可取 auto、cpu 或 cuda。示例是人工构造的峰表，用于演示格式和检查推理链；对应实跑输出见 [示例结果](examples/output.json)。不同硬件与 PyTorch 版本可能使接近的束搜索候选交换名次。

## 输入与输出

支持单个 JSON 对象、JSON 对象数组或 JSONL。每条记录包含可选的 sample_id、三个非空峰列表 nmr 和已知质量 known_mass；批内 ID 须唯一。

| 字段 | 含义 |
|---|---|
| nmr.1H、nmr.13C、nmr.31P | 数字表示单一化学位移（ppm）。峰对象可用 shift_ppm，或用 shift_min_ppm 和 shift_max_ppm 表示区间。 |
| 可选峰信息 | multiplicity 为文本，integral_protons 为非负积分，coupling_hz 为以 Hz 为单位的数值列表。缺失字段沿用训练时的缺失编码。 |
| known_mass | type 必须是 monoisotopic_mass，value_da 是以 Da 表示的中性单同位素质量，source 可记录来源。输入不接受平均分子量、未经换算的离子 m/z、真实分子式或参考 SMILES。 |

输出的 pool_size 是有效去重候选总数，mass_matches 是窗内候选数。每个候选列出最终名次、保留立体化学的规范 SMILES、精确质量、带符号的误差（ppm）和窗内标识。对给定质量 M 和候选质量 m，满足 |m−M|/M×10⁶≤5 的候选前移。窗内、窗外各自保持原顺序，无窗内候选时保留原排序。空池返回空列表。top-k 只控制展示条数。

## 组成与训练

| 组件 | 用途 |
|---|---|
| primary、secondary | 3 层编码器、4 层解码器，种子 42 和 7。各用束宽 10 和 30 生成候选。 |
| extra | 5 层编码器、4 层解码器，种子 42，使用束宽 10。 |
| ranker | 对冻结谱图特征、候选指纹和描述符进行相容性评分。 |
| element | 用谱图预测九类元素计数，计算组成一致性得分。 |

主、次序列分数权重为 0.6 和 0.4；标准化的兼容性和元素一致性项各用 0.25。评分后的池与额外来源交替合并并去重，再使用精确质量排序。词表、多重度映射、RBF 编码、元素头归一化参数及架构在 model_config.json 中。五个组件均以 safetensors 保存。加载时核验权重文件与张量哈希。

## 论文评估

| 2,222 条既有测试记录 | Top‑1 | Top‑5 | Top‑10 |
|---|---:|---:|---:|
| 相同候选池，关闭质量优先排序 | 26.06% | 43.38% | 50.18% |
| 开启已知理论精确质量优先排序 | **38.79%** | **55.27%** | **56.71%** |

质量开启后三项分别命中 862、1,228 和 1,260 条；候选池在 1,268 条记录中包含参考结构。基准质量与参考结构计算值一致，测试集曾在开发中被检视。上述结果对应已知理论质量条件。实测 HRMS 需另行处理电荷、加合物和测量误差。

## 文件与复现范围

- [验证与测试指标](evaluation/hcp_metrics.json)含本次发布系统的质量开关对照与生成结构的 R1/R2 指标。独立的磷中心分类探针属于另一实验，此仓库未打包其权重。
- [模型来源](evaluation/weight_provenance.json)给出五个原始检查点及张量的 SHA‑256。[划分清单](evaluation/split_ids.csv)仅包含内部记录 ID 和分区，训练、验证、测试分别为 17,782、2,222、2,222 条。
- [研究代码与复现说明](research/README.md)包括主六组实验的训练和评估实现。原始谱图、规范化训练集、增强缓存及分子库的再分发权限尚未明确，这些文件未随仓库发布。
- 论文训练环境为 Python 3.10.8、PyTorch 2.6.0+cu124、NumPy 2.2.6、RDKit 2025.09.4。发布转换对全部五份参数逐张量比对，并在本机 CPU 上使用同一合成输入与原始系统比对完整候选池和最终排序。

## 许可证

本仓库的 PTrans 代码和五份模型权重均采用 [MIT 许可证](LICENSE)。第三方依赖沿用各自许可证。
