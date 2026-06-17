# data/

本目录下的数据文件不入库（详见根目录 `.gitignore`），clone 后请按以下步骤复现：

## 1. 下载 BFCL v3 原始数据

```bash
python scripts/download_data.py
```

该脚本从 HuggingFace `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
拉取以下文件到 `data/`：

- 多轮：`BFCL_v3_multi_turn_{base,composite,long_context,miss_func,miss_param}.json`
- 单轮：`BFCL_v3_{simple,multiple,parallel,parallel_multiple}.json`
- ground truth：`data/possible_answer/*.json`
- 工具实现快照：`gorilla_file_system.json` 等

## 2. 生成 SchemaShift 扰动数据集

```bash
python scripts/generate_perturbations.py
```

输出：`data/bfcl_v3_perturbed.json`（约 130 MB，超过 GitHub 单文件 100 MB 限制，故不入库）。

## 3. 构建训练用 parquet

```bash
python scripts/build_parquet.py
```

输出：`data/*.parquet`（verl 训练直接消费）。
