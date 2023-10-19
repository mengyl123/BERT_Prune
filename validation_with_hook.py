from datasets import load_dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import DataCollatorWithPadding
import torch
import sys
import numpy as np
import time
import copy

datasets_name="sst2"

token_size=128#这个参数暂时没用
batch_size=16

features_in_hook = []
features_out_hook = []

#下面是两个hook机制，能对模型中间值操作，容易爆显存
def OutputHook(module, fea_in, fea_out):
#hook机制，能对指定层的输出进行操作并返回，return的值将作为指定层的输出。想对激活值操作就在这里改
    features_out_hook.append(fea_out)
    return None

def InputHook(module, fea_in, fea_out):
#hook机制，能对指定层的输入进行操作并返回，return的值将作为指定层的输入。想对激活值操作就在这里改
    features_in_hook.append(fea_in)
    return None


#计算最终模型输出的正确率
def flat_accuracy(preds, labels):
    #模型最终的输出是一组向量类型的数，要通过argmax输出label
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#从huggingface transformer库里加载模型相关的分词器（tokenizer）和模型主体（model），过程需联网
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2,output_hidden_states=True)#type:transformers.models.bert.modeling_bert.BertForSequenceClassification
model.load_state_dict(torch.load('./best_SST2.pth',map_location='cuda:0'))

#从huggingface transformer库里加载数据集，过程需联网
raw_datasets = load_dataset("glue", datasets_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#注册InputHook函数
h=list(torch.zeros(12))
for i in range(12):
     h[i]=model.bert.encoder.layer[i].output.register_forward_hook(hook=InputHook)

model.to(device)

#词表向量化函数，GLUE数据集的不同子集需要对应更改
def tokenize_function(example):
    return tokenizer(example["sentence"],
                     truncation=True,)

#对数据集操作，变成模型可以识别的格式
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence",  "idx"])#GLUE数据集的不同子集需要更改这句
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

#抽取测试集子集
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=3689).select(range(batch_size))#(seed=42).select(range(batch_size))

#dataloader类，drop_last代表扔掉最后一个batch，保持数据大小相同。collate_fn代表数据合并的格式，没有详细研究过怎么改
from torch.utils.data import DataLoader
validation_dataloader = DataLoader(
    small_eval_dataset,batch_size=batch_size, collate_fn=data_collator,drop_last=True
)
#这是整个测试集
# validation_dataloader = DataLoader(
#     tokenized_datasets["validation"],batch_size=batch_size, collate_fn=data_collator,drop_last=True
# )

counter=0
#模型切换到测试模式，不产生也不传播梯度
model.eval()
total_eval_accuracy = 0
for batch in validation_dataloader:
    #print(batch)
    batch = {k: v.to(device) for k, v in batch.items()}
        # 评估的时候不需要更新参数、计算梯度
    with torch.no_grad():
        outputs = model(**batch)
    b_labels = batch["labels"]
    # 将预测结果和 labels 加载到 cpu 中计算
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 计算准确率
    total_eval_accuracy += flat_accuracy(logits, label_ids)

    #读取attention模块的输出，这个方法是huggingface里bert类自带的
    hidden_states = outputs[2]
    #print(hidden_states.size())
    attention_hidden_states = hidden_states[1:]
    #读取FFN模块的输出，这个方法是在上面实现的
    for i in range(12):
        print((features_in_hook[i][0] == 0).sum())
    #attention_hidden_states[i]是第i层的attention层的输出
	#features_in_hook[i][0]是第i层的FFN层的输出
        #output_sum[i] += (attention_hidden_states[i] == 0).cpu().sum() / (batch_size * 128 * 768)
        #intermediate_sum[i] += (features_in_hook[i][0] == 0).cpu().sum() / (batch_size * 128 * 3072)
        #torch.save(attention_hidden_states[i],f"./activation_tensor/output/{datasets_name}_dense_{i}.pt")
        #torch.save(features_in_hook[i], f"./activation_tensor/intermediate/{datasets_name}_{sparsity}_{i}.pt")
        #attention_out_sum[i] += (features_in_hook[i] == 0).cpu().sum() / (batch_size * 128 * 768)
    features_in_hook.clear()
    counter+=1
#清空hook缓存，不然爆显存，因为用了append（）方法，features_in_hook会越来越多
for i in range(12):
    h[i].remove()
print(counter)
# cnt=0
# for i in features_in_hook:
#     cnt=cnt+1
# print(cnt)
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

