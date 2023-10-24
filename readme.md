# BERT-Based 推理和激活值稀疏化

使用huggingface的Transformer库实现的BERT-Based模型的推理和微调。推理脚本为[glue_validation.py](https://github.com/mengyl123/BERT_Prune/blob/main/glue_validation.py)，微调脚本为[train_on_GLUE.py](https://github.com/mengyl123/BERT_Prune/blob/main/train_on_GLUE.py)。Linux上运行脚本所需的库见[requirements.txt](https://github.com/mengyl123/BERT_Prune/blob/main/requirements.txt).
[validation_with_hook.py](https://github.com/mengyl123/BERT_Prune/blob/main/validation_with_hook.py)为带有hook机制的推理脚本，能够将模型中间层的值输出并进行修改。
[SparseLayer.py](https://github.com/mengyl123/BERT_Prune/blob/main/SparseLayer.py)为自定义的稀疏化层，可作为自定义层的修改示例。

## 运行脚本
无需命令行里附加参数，直接运行各个脚本即可

**加粗文本**

*斜体文本*

[链接文本](链接URL)

- 无序列表项
- 无序列表项
- 无序列表项

1. 有序列表项
2. 有序列表项
3. 有序列表项

> 引用文本

`代码块`

```python
print("Hello, World!")
