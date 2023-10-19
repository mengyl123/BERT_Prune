from datasets import load_dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
import sys
PATH='./prune&fix/best_SST2.pth'


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.load_state_dict(torch.load('best_SST2.pth'))
raw_datasets = load_dataset("glue", "sst2")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#print(raw_datasets)
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)#,padding="max_length",max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# print(raw_datasets)
# raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader

batch_size=128

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
validation_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
)
from transformers import AdamW

optimizer = AdamW(model.parameters(),
                  lr = 1e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )

from transformers import get_scheduler

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

import numpy as np
import copy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

training_stats = []
best_model=None
best_acc = 0

for epoch in range(num_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        #print(batch['attention_mask'].size())
        # print(batch['labels'].size())
        # print(batch['token_type_ids'].size())
        # print(batch['input_ids'].size())
        if step % 40 == 1 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
            #print(batch["labels"])
        model.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("")
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
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    # 统计本次 epoch 的 loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )
    if total_eval_accuracy > best_acc:
        best_model = copy.deepcopy(model)
        best_model = best_model.cpu()
        best_acc = total_eval_accuracy
torch.save(best_model.state_dict(), PATH)

print("")
print("Training complete!")

import pandas as pd

# 保留 2 位小数
pd.set_option("display.precision", 4)
# 加载训练统计到 DataFrame 中
df_stats = pd.DataFrame(data=training_stats)

# 使用 epoch 值作为每行的索引
df_stats = df_stats.set_index('epoch')

# 展示表格数据
print(df_stats)

