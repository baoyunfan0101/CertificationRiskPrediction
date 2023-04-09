import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression as LR
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
    LR_model = LR(
        penalty='l1',  # 惩罚项
        dual=False,  # 对偶(或原始)方法
        tol=0.0001,  # 停止求解的标准
        C=1.0,  # 正则化系数λ的倒数
        fit_intercept=True,  # 存在截距或偏差
        intercept_scaling=1,
        class_weight={0: 0.225, 1: 0.775},  # 分类模型中各类型的权重
        random_state=None,  # 随机种子
        solver='liblinear',  # 优化算法选择参数
        max_iter=10,  # 算法收敛最大迭代次数
        multi_class='auto',  # 分类方式选择参数
        verbose=0,  # 日志冗长度
        warm_start=False,  # 热启动参数
        n_jobs=None,  # 并行数
        l1_ratio=None
    )
    LR_model = LR_model.fit(X_train, Y_train)

    # 保存模型
    with open('LR_model.pkl', 'wb') as f:
        pickle.dump(LR_model, f)


# 本地测试
X_train, X_test, Y_train, Y_test = load_data()
trainModel(X_train, Y_train)

# 读取模型
with open('LR_model.pkl', 'rb') as file:
    LR_model = pickle.load(file)

# 评分
LR_model.predict(X_test)
accuracy = LR_model.score(X_test, Y_test)
print('正确率为%s' % accuracy)

# AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

score = LR_model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(Y_test, score)  # 计算真正率和假正率
roc_auc = auc(fpr, tpr)  # 计算auc的值
print('AUC为' + str(roc_auc))

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='red', lw=lw, label='LR(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
