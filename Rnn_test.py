import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys

MAX_SEQUENCE_LENGTH = 6269  # 最大序列长度
BATCH_SIZE = 20  # 批处理大小
CLASS_NUM = 2  # 类别数（输出类别数）
EMBEDDING_DIM = 64  # 嵌入维度
LEARNING_RATE = 0.001  # 学习率
HIDDEN_DIM = 128  # RNN隐藏层维度
NUM_LAYERS = 2  # RNN层数
DROPOUT_VALUE = 0.5  # Dropout比率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
MODEL_PATH = 'weight2class.h5'  
EPOCHS = 3  #

# 定义RNN模型
class EmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, class_size, num_layers=2, dropout_value=0.5):
        super(EmbeddingRNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 嵌入层
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_value, batch_first=True)  # LSTM层
        self.fc = nn.Linear(hidden_dim, class_size)  # 全连接层
        self.dropout = nn.Dropout(dropout_value)  # Dropout层

    def forward(self, x):
        x = self.embeddings(x)  # 输入经过嵌入层
        x, _ = self.rnn(x)  # 输入经过RNN层
        x = self.dropout(x[:, -1, :])  # 取最后一个时间步的输出，并应用Dropout
        out = self.fc(x)  # 输入经过全连接层
        return out

def load_model(vocab_size, embedding_dim, hidden_dim, class_size, model_path, device):
    model = EmbeddingRNN(vocab_size, embedding_dim, hidden_dim, class_size)  # 初始化模型
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载预训练模型参数
    return model.to(device)  # 将模型移动到指定设备

# 获取预测概率
@torch.no_grad()
def get_probas(model, dataloader):
    model.eval()  # 设置模型为评估模式
    scores = []
    F_softmax = torch.nn.Softmax(dim=1)  # 定义Softmax层
    for x, _ in dataloader:
        x = x.long().to(DEVICE)  # 将输入数据移动到设备
        y_hat = model(x)  # 模型前向传播
        scores.append(F_softmax(y_hat.cpu()).numpy())  # 计算并保存Softmax概率
    return np.concatenate(scores)  # 返回拼接后的概率数组

# 预测类别
def predict(model, dataloader):
    probas = get_probas(model, dataloader)  # 获取预测概率
    prediction_list = np.argmax(probas, axis=1)  # 获取最大概率对应的类别
    return prediction_list  # 返回预测类别列表

# 保存预测结果
def save_predictions(predictions, output_file="cnn_2class.out"):
    clsid_to_class_name = {0: '0', 1: '1'}  # 类别ID到类别名的映射
    pred_label = [clsid_to_class_name[i] for i in predictions]  # 将类别ID转换为类别名
    df = pd.DataFrame(pred_label)  # 创建DataFrame
    df.to_csv(output_file, sep=" ", index=False, header=False)  # 保存到文件

# 主函数
def main(input_tensor_file, output_file, model_path):
    # 加载输入数据
    fake_data = torch.load(input_tensor_file)
    query_dataset = TensorDataset(fake_data, torch.zeros(len(fake_data)))
    query_dataloader = DataLoader(query_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 加载预训练模型
    model = load_model(MAX_NB_CHARS, EMBEDDING_DIM, HIDDEN_DIM, CLASS_NUM, model_path, DEVICE)
    
    predictions = predict(model, query_dataloader)
    
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_tensor_file> <output_file>")
        sys.exit(1)
    
    input_tensor_file = sys.argv[1]  # 输入张量文件
    output_file = sys.argv[2]  
    
    main(input_tensor_file, output_file, MODEL_PATH)  
