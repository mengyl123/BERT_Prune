from datasets import load_dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
import sys

datasets_name="sst2"

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

#path换成best_SST2.pth的路径
#model.load_state_dict(torch.load(f'path'))

raw_datasets = load_dataset("glue", datasets_name)
print("datasets contain:\n",raw_datasets)

#sys.exit(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def tokenize_function(example):
    return tokenizer(example["sentence"],
                     example["label"],
                     truncation=True,
                     )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence","idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader

batch_size=16

validation_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
)


import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


print("Running Validation...")

model.eval()

# Tracking variables
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

# Evaluate data for one epoch
for batch in validation_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    # 评估的时候不需要更新参数、计算梯度
    with torch.no_grad():
        outputs = model(**batch)
    # 累加 loss
    total_eval_loss += outputs.loss.item()

    b_labels=batch["labels"]
    # 将预测结果和 labels 加载到 cpu 中计算
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 计算准确率
    total_eval_accuracy += flat_accuracy(logits, label_ids)

# 打印本次 epoch 的准确率
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
# 统计本次 epoch 的 loss
avg_val_loss = total_eval_loss / len(validation_dataloader)
print("  Validation Loss: {0:.2f}".format(avg_val_loss))
