import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

'''
# 读取数据
data = pd.read_excel('predict.xlsx', sheet_name='Sheet1')
'''

'''
# 读取模型
with open('LR_model.pkl', 'rb') as file:
    LR_model = pickle.load(file)
# 预测
score = LR_model.predict(data)
'''

'''
# 读取模型
with open('XGB_model.pkl', 'rb') as file:
    XGB_model = pickle.load(file)
# 预测
score = XGB_model.predict(data)
'''

'''
# 读取模型
with open('SVM_model.pkl', 'rb') as file:
    SVM_model = pickle.load(file)
# 预测
score = SVM_model.predict(data)
'''

# 读取数据
data = pd.read_excel('predict.xlsx', sheet_name='Sheet1')
data = data.drop(
    ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
     'op_target_1'], axis=1)
scalar = StandardScaler().fit(data)
X = scalar.transform(data)

# 读取模型
with open('OneClassSVM_model.pkl', 'rb') as file:
    OneClassSVM_model = pickle.load(file)
# 预测
score = OneClassSVM_model.predict(X)

with open('ret.csv', 'w', encoding='utf-8') as f:
    f.write('id,ret\n')
    for idx in range(len(score)):
        f.write(str(idx + 1) + ',' + str(0 if score[idx] == -1 else 1) + '\n')
    f.close()
