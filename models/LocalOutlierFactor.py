import pandas as pd
import pickle
from sklearn.neighbors import LocalOutlierFactor
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

    # 读取predict.xlsx
    data = pd.read_excel('predict.xlsx', sheet_name='Sheet1')
    data = data.drop(
        ['Unnamed: 0', 'client_type', 'browser_source', 'ip_location_type_keyword_2', 'ip_location_type_keyword_4',
         'op_target_1'], axis=1)
    scalar = StandardScaler().fit(data)
    X_test = scalar.transform(data)
    return X_train, _X_train, X_test


def trainModel(X_train, _X_train, X_test):
    # 训练模型(半监督)
    LOF_model = LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        contamination='auto',
        novelty=True,  # 半监督学习
        n_jobs=None
    )
    LOF_model = LOF_model.fit(_X_train)

    # 保存模型
    with open('LOF_model.pkl', 'wb') as f:
        pickle.dump(LOF_model, f)

    # 训练模型(无监督)
    LOF_model = LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        contamination=0.5,
        novelty=False,  # 无监督学习
        n_jobs=None
    )
    score = LOF_model.fit_predict(X_test)

    # 输出结果
    with open('ret.csv', 'w', encoding='utf-8') as f:
        f.write('id,ret\n')
        for idx in range(len(score)):
            f.write(str(idx + 1) + ',' + str(0 if score[idx] == -1 else 1) + '\n')
        f.close()


# 学习
X_train, _X_train, X_test = load_data()
trainModel(X_train, _X_train, X_test)
