import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import sys

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 配置参数
MAX_SEQUENCE_LENGTH = 6269
BATCH_SIZE = 20
CLASS_NUM = 26
EMBEDDING_DIM = 64
LEARNING_RATE = 0.002
HIDDEN_DIM = 128  # RNN隐藏层维度
NUM_LAYERS = 2  # RNN层数
DROPOUT_VALUE = 0.5
MODEL_PATH = '3_27classes.h5'

# RNN模型定义
class EmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, class_size, num_layers=2, dropout_value=0.5):
        super(EmbeddingRNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_value, batch_first=True)
        self.fc = nn.Linear(hidden_dim, class_size)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])  # 使用最后一个时间步的输出
        out = self.fc(x)
        return out

def load_model(vocab_size, embedding_dim, hidden_dim, class_size, model_path, device):
    model = EmbeddingRNN(vocab_size, embedding_dim, hidden_dim, class_size, num_layers=NUM_LAYERS, dropout_value=DROPOUT_VALUE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)

@torch.no_grad()
def get_probas(model, dataloader):
    model.eval()
    scores = []
    F_softmax = torch.nn.Softmax(dim=1)
    for x, _ in dataloader:
        x = x.long().to(device)
        y_hat = model(x)
        scores.append(F_softmax(y_hat.cpu()).numpy())
    return np.concatenate(scores)

def save_predictions(probas, output_file="N_rnn.out"):
    df = pd.DataFrame(probas)
    df.to_csv(output_file, sep=" ", index=False)

def main(input_tensor_file, output_file, model_path):
    # 加载数据
    fake_data = torch.load(input_tensor_file)
    query_dataset = TensorDataset(fake_data, torch.zeros(len(fake_data)))
    query_dataloader = DataLoader(query_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 加载模型
    model = load_model(MAX_NB_CHARS, EMBEDDING_DIM, HIDDEN_DIM, CLASS_NUM, model_path, device)
    
    # 预测
    probas = get_probas(model, query_dataloader)
    
    # 保存预测结果
    save_predictions(probas, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_tensor_file> <output_file>")
        sys.exit(1)
    
    input_tensor_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_tensor_file, output_file, MODEL_PATH)
