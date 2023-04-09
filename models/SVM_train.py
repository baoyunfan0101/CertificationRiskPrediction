import pandas as pd
import pickle
from sklearn import svm
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
    SVM_model = svm.SVC(
        C=0.5,  # 惩罚参数
        kernel='linear',  # 核函数
        degree=3,  # 多项式维度
        gamma='auto',
        coef0=0.0,  # 核函数常数项
        shrinking=True,  # 采用shrinking heuristic方法
        probability=True,  # 采用概率估计
        tol=0.001,  # 停止训练的误差值
        cache_size=200,  # 核函数cache缓存
        class_weight=None,  # 类别权重
        verbose=False,  # 允许冗余输出
        max_iter=-1,  # 最大迭代次数无约束
        decision_function_shape='ovo',  # 多分类策略
        random_state=None,  # 随机种子
    )
    SVM_model = SVM_model.fit(X_train, Y_train)

    # 保存模型
    with open('SVM_model.pkl', 'wb') as f:
        pickle.dump(SVM_model, f)


# 本地测试
X_train, X_test, Y_train, Y_test = load_data()
trainModel(X_train, Y_train)

# 读取模型
with open('SVM_model.pkl', 'rb') as file:
    SVM_model = pickle.load(file)

# 评分
SVM_model.predict(X_test)
accuracy = SVM_model.score(X_test, Y_test)
print('正确率为%s' % accuracy)

# AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

score = SVM_model.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(Y_test, score)  # 计算真正率和假正率
roc_auc = auc(fpr, tpr)  # 计算auc的值
print('AUC为' + str(roc_auc))

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='red', lw=lw, label='SVM(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()