import pandas as pd
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def load_data():
    # 读取OneClasstrain.xlsx
    data = pd.read_excel('OneClasstrain.xlsx', sheet_name='Sheet1')
    data = data.drop(
        ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
         'op_target_1'], axis=1)
    scalar = StandardScaler().fit(data)
    X_train = scalar.transform(data)

    # 读取_OneClasstrain.xlsx
    data = pd.read_excel('_OneClasstrain.xlsx', sheet_name='Sheet1')
    data = data.drop(
        ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
         'op_target_1'], axis=1)
    scalar = StandardScaler().fit(data)
    _X_train = scalar.transform(data)
    return X_train, _X_train


def trainModel(X_train, _X_train):
    # 训练模型
    OneClassSVM_model = OneClassSVM(
        kernel='rbf',  # 核函数
        degree=3,  # 多项式维度
        gamma='auto',
        coef0=0.0,  # 核函数常数项
        shrinking=True,  # 采用shrinking heuristic方法
        tol=0.001,  # 停止训练的误差值
        cache_size=200,  # 核函数cache缓存
        verbose=False,  # 允许冗余输出
        max_iter=-1,  # 最大迭代次数无约束
    )
    OneClassSVM_model = OneClassSVM_model.fit(_X_train)

    # 保存模型
    with open('OneClassSVM_model.pkl', 'wb') as f:
        pickle.dump(OneClassSVM_model, f)


# 学习
X_train, _X_train = load_data()
trainModel(X_train, _X_train)