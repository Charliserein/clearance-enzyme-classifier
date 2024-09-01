import pandas as pd
import xgboost as xgb
import joblib

def predict_and_save_results(model_path, input_data_path, output_txt_path, output_csv_path):

    
    clf = joblib.load(model_path)
    data1 = pd.read_csv(input_data_path, sep='\t', header=0)
    
    X = data1.drop(columns=['#'])
    
    dtest = xgb.DMatrix(X)
    
    y_pred = clf.predict(dtest)
    
    f2 = pd.DataFrame(y_pred)
    
    X = data1['#']
    x = pd.DataFrame(X)
    
    a3 = pd.concat([x, f2], axis=1)
    a3.to_csv(output_txt_path, sep=" ", index=False)
    
    f3 = pd.DataFrame(f2.idxmax(1))
    
    # 将 "#" 列与预测的类别索引拼接，并保存为CSV文件
    a4 = pd.concat([x, f3], axis=1)
    a4.to_csv(output_csv_path, sep="\t", header=None, index=False)

# 示例调用函数
predict_and_save_results('xgboos_Nclass.pkl', 'CKSAAGP.out', 'xgb_class.out', 'xgb_class.csv')
