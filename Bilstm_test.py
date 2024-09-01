import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 乘以2因为是双向LSTM

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 取最后一个时间步的输出
        return out

# 加载模型
def load_model(model_path, input_size, hidden_size, output_size):
    model = BiLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

# 加载数据
def load_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
        for line in lines:
            items = line.strip().split("\t")
            name = items[0]
            features = [float(item) for item in items[1:]]
            data[name] = features
    return data

# 进行预测
def predict(model, data):
    tensor_data = torch.tensor(list(data.values()), dtype=torch.float32)
    with torch.no_grad():  # 关闭梯度计算
        output = model(tensor_data)
        predictions = torch.argmax(output, dim=1).tolist()
    return predictions

# 显示结果
def display_results(predictions, data):
    for i, name in enumerate(data.keys()):
        print(f"样本 {name} 的预测结果为: {predictions[i]}")

def main(model_path, data_file):
    # 参数配置
    input_size = 20  # 假设特征维度为20
    hidden_size = 64
    output_size = 2  # 二分类任务

    # 加载模型
    model = load_model(model_path, input_size, hidden_size, output_size)
    
    # 加载数据
    data = load_data(data_file)
    
    # 进行预测
    predictions = predict(model, data)
    
    # 显示结果
    display_results(predictions, data)

if __name__ == "__main__":
    model_path = 'lstm_class.pkl'  # 模型路径
    data_file = 'DPC.out'  # 数据文件路径
    main(model_path, data_file)
