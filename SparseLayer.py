from torch import nn
#from torch import *
import torch
class SparseBertSelfOutput(nn.Module):
    def __init__(self, Th):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.Th=Th

    def activation_sparse(self,hidden_states,Th):
        mask=hidden_states.data.abs()>Th
        pruning_param = hidden_states * mask
        return pruning_param

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.activation_sparse(hidden_states, self.Th)
        return hidden_states

class SparseOutput(nn.Module):
    def __init__(self,Th):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.Th = Th

    def activation_sparse(self,hidden_states,Th):
        mask=hidden_states.data.abs()>Th
        pruning_param = hidden_states * mask
        return pruning_param

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states=self.activation_sparse(hidden_states,self.Th)
        return hidden_states

class SparseIntermediate(nn.Module):
    def __init__(self,Th):
        super().__init__()
        self.dense = nn.Linear(768, 3072)
        self.intermediate_act_fn = nn.functional.gelu
        self.Th = Th

    def activation_sparse(self,hidden_states,Th):
        mask=hidden_states.data.abs()>Th
        pruning_param = hidden_states * mask
        return pruning_param

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.activation_sparse(hidden_states, self.Th)
        return hidden_states

def SparseBertSelfOutputInit(model,threshold):
    temp_dense_weight=[]
    temp_dense_bias=[]
    temp_LayerNorm_weight=[]
    temp_LayerNorm_bias=[]
    for i in range(12):
        temp_dense_weight.append(model.bert.encoder.layer[i].attention.output.dense.weight)
        temp_dense_bias.append(model.bert.encoder.layer[i].attention.output.dense.bias)
        temp_LayerNorm_weight.append(model.bert.encoder.layer[i].attention.output.LayerNorm.weight)
        temp_LayerNorm_bias.append(model.bert.encoder.layer[i].attention.output.LayerNorm.bias)
    for i in range(12):
        model.bert.encoder.layer[i].attention.output = SparseBertSelfOutput(threshold[i])
        model.bert.encoder.layer[i].attention.output.dense.weight = temp_dense_weight[i]
        model.bert.encoder.layer[i].attention.output.dense.bias = temp_dense_bias[i]
        model.bert.encoder.layer[i].attention.output.LayerNorm.weight = temp_LayerNorm_weight[i]
        model.bert.encoder.layer[i].attention.output.LayerNorm.bias = temp_LayerNorm_bias[i]

def SparseOutputInit(model,threshold):
    temp_dense_weight=[]
    temp_dense_bias=[]
    temp_LayerNorm_weight=[]
    temp_LayerNorm_bias=[]
    for i in range(12):
        temp_dense_weight.append(model.bert.encoder.layer[i].output.dense.weight)
        temp_dense_bias.append(model.bert.encoder.layer[i].output.dense.bias)
        temp_LayerNorm_weight.append(model.bert.encoder.layer[i].output.LayerNorm.weight)
        temp_LayerNorm_bias.append(model.bert.encoder.layer[i].output.LayerNorm.bias)
    for i in range(12):
        model.bert.encoder.layer[i].output = SparseOutput(threshold[i])
        model.bert.encoder.layer[i].output.dense.weight = temp_dense_weight[i]
        model.bert.encoder.layer[i].output.dense.bias = temp_dense_bias[i]
        model.bert.encoder.layer[i].output.LayerNorm.weight = temp_LayerNorm_weight[i]
        model.bert.encoder.layer[i].output.LayerNorm.bias = temp_LayerNorm_bias[i]

def SparseIntermediateInit(model,threshold):
    temp_dense_weight = []
    temp_dense_bias = []
    for i in range(12):
        temp_dense_weight.append(model.bert.encoder.layer[i].intermediate.dense.weight)
        temp_dense_bias.append(model.bert.encoder.layer[i].intermediate.dense.bias)
    for i in range(12):
        model.bert.encoder.layer[i].intermediate = SparseIntermediate(threshold[i])
        model.bert.encoder.layer[i].intermediate.dense.weight = temp_dense_weight[i]
        model.bert.encoder.layer[i].intermediate.dense.bias = temp_dense_bias[i]