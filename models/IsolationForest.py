import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def load_data():
    # 读取train.xlsx
    data = pd.read_excel('OneClasstrain.xlsx', sheet_name='Sheet1')
    data = data.drop(
        ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
         'op_target_1'], axis=1)
    scalar = StandardScaler().fit(data)
    X_train = scalar.transform(data)

    # 读取predict.xlsx
    data = pd.read_excel('predict.xlsx', sheet_name='Sheet1')
    data = data.drop(
        ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
         'op_target_1'], axis=1)
    scalar = StandardScaler().fit(data)
    X_test = scalar.transform(data)
    return X_train, X_test


def trainModel(X_train, X_test):
    # 训练模型
    IsolationForest_model = IsolationForest(
        n_estimators=100,  # 随机树数量
        max_samples='auto',  # 子采样的大小
        contamination=0.5,  # 异常数据占给定的数据集的比例
        max_features=1.0,  # 每棵树iTree的属性数量
        bootstrap=False,  # 执行不放回的采样
        n_jobs=-1,  # 并行运行的作业数量
        random_state=None,  # 随机数生成器
        verbose=0,  # 打印日志的详细程度
        warm_start=False  # fit时不重用上一次调用的结果
    )
    IsolationForest_model.fit(X_train)
    score = IsolationForest_model.predict(X_test)

    # 输出结果
    with open('ret.csv', 'w', encoding='utf-8') as f:
        f.write('id,ret\n')
        for idx in range(len(score)):
            f.write(str(idx + 1) + ',' + str(0 if score[idx] == -1 else 1) + '\n')
        f.close()

# 学习
X_train, X_test = load_data()
trainModel(X_train, X_test)
