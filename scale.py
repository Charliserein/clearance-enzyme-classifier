import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(file_path, sep="\t"):

    data = pd.read_csv(file_path, sep=sep)
    data = data.drop(columns=['#'])  # 删除指定列
    return data

def scale_data(data, feature_range=(0, 1)):
    """
    对数据进行归一化处理。
    scaled_data: 归一化处理后的数据
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data)

def save_data(data, output_file, sep="\t"):

    data.to_csv(output_file, sep=sep, index=False)

def main(input_file, output_file):

    data = load_data(input_file)  # 加载数据
    scaled_data = scale_data(data)  # 归一化数据
    save_data(scaled_data, output_file)  # 保存结果

if __name__ == "__main__":
    input_file = "01"  # 输入文件路径
    output_file = "DPC.out"  # 输出文件路径
    main(input_file, output_file)  # 执行主函数
