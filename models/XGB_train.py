import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def load_data():
    # 读取train.xlsx
    data = pd.read_excel('train.xlsx', sheet_name='Sheet1')
    data = data.drop('Unnamed: 0', axis=1)
    X = data.drop('risk_label', axis=1)
    Y = data['risk_label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    return X_train, X_test, Y_train, Y_test


def trainModel(X_train, Y_train):
    # 训练模型
    XGB_model = XGBClassifier(
        max_depth=6,  # 树的深度
        learning_rate=0.1,  # 学习率
        n_estimators=100,  # 迭代次数(决策树个数)
        silent=False,  # 是否输出中间过程
        objective='binary:logitraw',  # 目标函数
        booster='gbtree',  # 基分类器
        nthread=-1,  # 使用全部CPU进行并行运算
        gamma=10,  # 惩罚项系数
        min_child_weight=1,  # 最小叶子节点样本权重和
        max_delta_step=0,  # 对权重改变的最大步长无约束
        subsample=1,  # 每棵树训练集占全部训练集的比例
        colsample_bytree=1,  # 每棵树特征占全部特征的比例
        colsample_bylevel=1,
        eta=0.1,
        reg_alpha=0,  # L1正则化系数
        reg_lambda=1,  # L2正则化系数
        scale_pos_weight=5,  # 正样本的权重
        base_score=0.5,
        random_state=0,
        seed=None,  # 随机种子
        missing=None,
        use_label_encoder=False
    )
    XGB_model = XGB_model.fit(X_train, Y_train)

    # 保存模型
    with open('XGB_model.pkl', 'wb') as f:
        pickle.dump(XGB_model, f)


# 本地测试
X_train, X_test, Y_train, Y_test = load_data()
trainModel(X_train, Y_train)

# 读取模型
with open('XGB_model.pkl', 'rb') as file:
    XGB_model = pickle.load(file)

# 评分
XGB_model.predict(X_test)
accuracy = XGB_model.score(X_test, Y_test)
print('正确率为%s' % accuracy)

# AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

score = XGB_model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(Y_test, score)  # 计算真正率和假正率
roc_auc = auc(fpr, tpr)  # 计算auc的值
print('AUC为'+str(roc_auc))

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='red', lw=lw, label='XGBoost(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()