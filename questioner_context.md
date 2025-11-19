# Questioner Rollout with External Corpus

本文档记录了为 Questioner 训练加入语料上下文所做的全部代码修改，便于后续审查与复现。

## 1. 数据配置新增 `context_key`

- 文件：`verl/trainer/config.py`  
  - 位置：`DataConfig` 数据类（约第 34 行）  
  - 修改：新增 `context_key: Optional[str] = None` 字段，允许在配置中声明额外的上下文字段。

## 2. Dataloader 传递 `context_key`

- 文件：`verl/trainer/data_loader.py`  
  - 位置：`create_dataloader`（约第 26-75 行）  
  - 修改：在构建训练/验证 `RLHFDataset` 时将 `context_key=config.context_key` 传入，使数据集能够读取该字段。

## 3. `RLHFDataset` 中注入语料

- 文件：`verl/utils/dataset.py`
  1. 构造函数中新增参数 `context_key` 并保存为 `self.context_key`（约第 90-120 行）。  
  2. `_build_messages` 中读取 `example[self.context_key]`，并通过 `_append_context` 辅助函数把语料插入 questioner 的 user 提示（约第 137-214 行）。若 `context_key` 未配置或为空，则维持原提示不变。

## 4. Questioner 训练脚本接收语料路径

- 文件：`scripts/questioner_train.sh`
  - 修改点：
    - 新增第四个参数 `dataset_path`，若缺失则提示用法并退出（脚本顶部约第 1-10 行）。
    - 调用 `verl.trainer.main` 时增加：
      ```
      data.train_files=$dataset_path
      data.val_files=$dataset_path
      data.prompt_key=text
      data.context_key=text
      data.answer_key=id
      ```
      （命令行参数区域约第 25-40 行）

- 文件：`scripts/questioner_train_penalty.sh`
  - 与上同样的改动：新增 `dataset_path` 参数及同一组 `data.*` 覆盖项（脚本顶部与命令行参数部分）。

## 5. 使用方式

运行 questioner 训练时，新增的语料路径参数指向 parquet（或其他 RLHF 数据）文件，例如：

```bash
bash scripts/questioner_train_penalty.sh \
  <solver_model> \
  <questioner_model> \
  <save_name> \
  /Users/xiexuyan/Desktop/R-Zero/part_000000.parquet
```

脚本会自动设置 `prompt_key=context_key=text`，因此每条样本的 `text` 列既作为需基于的语料，也作为输入 prompt。本轮 rollout questioner 生成的题目将始终参考该内容。该记录可作为未来迭代或回滚的文档依据。 
