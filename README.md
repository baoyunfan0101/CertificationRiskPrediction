# Abnormal Accounts Recognition

## 关联项目

https://github.com/baoyunfan0101/AbnormalAccountsRecognition

## 文件说明

datasets // 数据集（训练集、测试集）  
feature engineering // 特征工程  
models // 评估模型

## 测试环境

Python3.8

## 任务描述

项目来自系统认证风险预测（https://www.datafountain.cn/competitions/537）。

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/problem.png)

本赛题中，参赛团队将基于用户认证行为数据及风险异常标记结构，构建用户认证行为特征模型和风险异常评估模型，利用风险评估模型去判断当前用户认证行为是否存在风险。

- 利用用户认证数据构建行为基线；
- 采用监督学习模型，基于用户认证行为特征，构建风险异常评估模型，判断当前用户认证行为是否存在风险。

## 特征工程

### 数据预处理

原始数据共包含18个特征、1个标签。但很多特征为离散的信息，不具备学习的价值，因而下面将分别阐述对每个特征采取的预处理方法。

---

- session_id（认证ID）  
检查是否存在重复的认证ID，将重复的行删去后，舍弃该特征。
- op_date（认证时间）  
先查找用户名和IP地址都相同的记录，然后对于每条记录，从上述用户名和IP地址都相同的记录中找出认证时间在该条记录前的最近一条记录，计算其认证时间差，作为新的特征“op_timedelta（认证时间差）”插入训练集，舍弃原始的认证时间。
- user_name（用户名）  
无学习价值，舍弃。
- action（操作类型）  
有“login”和“sso”两种可能的取值，分别替换为0和1。
- auth_type（首次认证方式）  
有“pwd”、“sms”、“otp”、“qr”和“（无）”五种可能的取值，因此对该特征进行one-hot编码，转化为5个布尔型的离散特征。特别地，其中“（无）”对于预测值的影响非常大，又因为前四个特征可能在后续步骤中被舍弃，因此将“（无）”单独作为一个特征。
- ip（IP地址）  
无学习价值，舍弃。
- ip_location_type_keyword（IP类型）  
有“家庭宽带”、“代理IP”、“内网”和“公共宽带”四种可能的取值，因此对该特征进行one-hot编码，转化为4个布尔型的离散特征。
- ip_risk_level（IP威胁级别）  
有“1级”、“2级”和“3级”三种可能的取值，由于三种取值间存在大小关系，因此分别将其替换为1、2和3。
- location（地点）  
学习的意义较小，舍弃。
- client_type（客户端类型）  
有“app”和“web”两种可能的取值，分别替换为0和1。
- browser_source（浏览器来源）  
有“desktop”和“mobile”两种可能的取值，分别替换为0和1。
- device_model（设备型号）  
学习的意义较小，舍弃。
- os_type（操作系统类型）  
有“windows”和“macOS”两种可能的取值，分别替换为0和1。
- os_version（操作系统版本号）  
学习的意义较小，舍弃。
- browser_type（浏览器类型）  
有“edge”、“chrome”、“firefox”、“ie”和“safari”五种可能的取值，因此对该特征进行one-hot编码，转化为5个布尔型的离散特征。
- browser_version（浏览器版本号）  
学习的意义较小，舍弃。
- bus_system_code（应用系统编码）  
有“attendance”、“coremail”、“crm”、“oa”、“order-mgnt”、“reimbursement”和“salary”七种可能的取值，因此对该特征进行one-hot编码，转化为7个布尔型的离散特征。
- op_target（应用系统类目）  
有“sales”、“finance”、“management”和“hr”四种可能的取值，因此对该特征进行one-hot编码，转化为4个布尔型的离散特征。

---

通过上述预处理过程，原始数据中的18个特征被编码为31个新特征。其中训练集数据带标签“risk_label（风险标识）”。

由于不同特征数据的量纲不一致，存在超出取值范围的离群数据，因此需进行数据标准化。这里基于原始数据的均值和标准差进行z-score标准化，以满足下列模型训练的需要，公式为

$$
{X'}_{i} = \frac{X_{i} - {\overset{-}{X}}_{i}}{S}
$$

其中
${X'}\_{i}$
为数据标准化后的特征；
$X_{i}$
原数据的特征；
${\overset{-}{X}}\_{i}$
为原数据特征的均值；S为原数据特征的标准差，其计算公式为
$\sqrt{\frac{\sum\limits_{i = 1}^{n}\left( {x_{i} - \overset{-}{x}} \right)^{2}}{n - 1}}$
。

*数据预处理的Python脚本在“preprocessing.py”中。*

### 特征的衍生和筛选

同一账户的操作和交易信息显然是账户特征模型的重点，而其操作和交易的时间信息（对应属性“tm_diff”）更是建立模型的重中之重。为此，我们参考RFM分析方法，对相关时间信息进行特征衍生。

RFM分析方法中的“RFM”分别指的是Recency（距离最近一次交易）、Frequency（交易频率）和Monetary（交易金额）。参考此方法的基本思想，我们从账户操作信息中提取出四个特征，分别为最近操作时间“op_recent_tm”、操作频次“op_frequency”、操作平均间隔“op_interval”和操作最小间隔“op_min_interval”；从账户交易信息中提取出五个特征，分别为最近交易时间“trans_recent_tm”、交易频次“trans_frequency”、交易金额“trans_amount”、交易平均间隔“trans_interval”和交易最小间隔“trans_min_interval”。

其中，同时在特征中保留平均间隔与最小间隔有特别的考虑。一方面，从专业角度来说，操作和交易的最小间隔是判断账户是否为人工处理的重要标准，对账户异常的识别有着特殊的价值；另一方面，平均间隔仅与账户的最早和最晚一次的操作或交易有关，而加入最小间隔能够更有效地利用数据，更完整地反映RFM分析方法中Frequency的概念。

特征的筛选过程中，除删除在上述“数据质量分析及数据预处理”部分提及的缺失值过多的属性外，还依据下面特征分析的结果进行了进一步地筛选，下面将会详细阐述。

*测试集和训练集的特征衍生也在“preprocessing_train.py”和“preprocessing_test.py”中，与数据预处理同步进行。测试集和训练集特征筛选的Python脚本分别在“screening_train.py”和“screening_test.py”中。*

### 特征分析

*特征分析的Python脚本在“iv.py”中。*

#### 特征重要性评估

**WOE**（Weight of Evidence，证据权重），是对原始自变量的一种编码形式，在对某个评价指标进行分组、离散化处理后，由下面公式计算

$$
{WOE}_{i} = ln\left( \frac{{py}_{i}}{{pn}_{i}} \right) = ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

其中
${WOE}\_{i}$
为第i组的WOE；
${py}\_{i}$
为第i组响应客户（即该问题中的风险账户）占所有样本中响应客户的比例；
${pn}\_{i}$
为第i组未响应客户占所有样本中未响应客户的比例；
$y_{i}$
为第i组响应客户的数量；
$y_{T}$
为第i组未响应客户的数量；
$n_{i}$
为所有样本中响应客户的数量；
$n_{T}$
为所有样本中未响应客户的数量。

**IV**（Information Value，信息价值），综合考虑了每组样本的WOE以及其在总体样本中所占的比例，可以看作WOE的加权和，在该问题中能够反映某一特征对账户风险的贡献率。某一组IV的具体计算公式为

$$
{IV}_{i} = \left( {py}_{i} - {pn}_{i} \right) \times {WOE}_{i} = \left( \frac{y_{i}}{y_{T}} - \frac{n_{i}}{n_{T}} \right) \times ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

其中
${IV}\_{i}$
为第i组的IV。某个特征IV的计算公式为

$$
IV = {\sum\limits_{i = 1}^{n}{IV}_{i}}
$$

其中n为组数。

考察上述数据预处理中得到的31个特征，除仅有0和1两种取值的布尔型离散特征外，分别以其它的每个特征为标准进行卡方分箱，将所有数据划分为2组（特别地，对于“op_timedelta（认证时间差）”特征，将所有数据划分为5组），再计算其IV，结果如下表。

特征|IV(*1e-4)
:---:|:---:
action|2.5060
ip_risk_level|6.3596
client_type|0.0
browser_source|0.0
os_type|1.3196
op_timedelta|22.1617
auth_type_1|0.9626
auth_type_2|0.6658
auth_type_3|0.7963
auth_type_4|2.8360
auth_type_5|2.5060
ip_location_type_keyword_1|5.4948
ip_location_type_keyword_2|0.1149
ip_location_type_keyword_3|4.5674
ip_location_type_keyword_4|0.0
browser_type_1|1.4198
browser_type_2|7.5429
browser_type_3|1.3926
browser_type_4|1.5188
browser_type_5|1.3196
bus_system_code_1|2.9250
bus_system_code_2|3.3532
bus_system_code_3|1.3104
bus_system_code_4|1.8696
bus_system_code_5|0.5283
bus_system_code_6|1.7523
bus_system_code_7|1.6988
op_target_1|0.0553
op_target_2|1.7523
op_target_3|6.2493
op_target_4|4.6095

观察各特征的IV，容易发现，“client_type（客户端类型）”、“browser_source（浏览器来源）”、“ip_location_type_keyword_2（IP类型为代理IP）”、“ip_location_type_keyword_4（IP类型为公共宽带）”和“op_target_1（应用系统类目为sales）”这5个特征对于“op_timedelta（认证时间差）”的重要性较低，因此在后续建模过程中会根据情况舍弃这5个特征。

#### 特征相关性分析

计算上述各评价指标的相关系数矩阵，并绘制热力图，如下图所示。

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/correlation.png)

从热力图中可知，除去上面提到的被舍弃的5个特征外，其它特征之间的相关性基本都在合理范围内，不存在相关性过大的特征。从这里也能侧面印证数据预处理过程中特征提取的合理性。

## 模型训练与优化

### 经典监督学习模型

在风险行为检测模型中，设因变量“risk_label（风险标识）”为y，其仅有1和0两个取值，可以看作二分类问题。

#### 逻辑回归

**逻辑回归**（Logistic Regression，LR）是一种广义的线性回归分析模型，常用于解决二分类问题。

若在自变量x=X的条件下因变量y=1的概率为p，记作
$p = P\left( y = 1 \middle| X \right)$
，则y=0的概率为
$1 - p$
，将因变量取1和0的概率比值
$\frac{p}{1 - p}$
记为优势比，对优势比取自然对数，即可得到Sigmoid函数

$$
Sigmoid(p) = ln\left( \frac{p}{1 - p} \right)
$$

令
$Sigmoid(p) = z$
，则有

$$
p = \frac{1}{1 + e^{- z}}
$$

设各特征的向量为X，系数向量为β，代入上式的z中，即得到回归模型的表达式

$$
h(x) = \frac{1}{1 + e^{- X\beta^{T}}}
$$

其中，h(x)的取值范围为[0,1]，可以表示题目所需的账户风险“label”的预测值。又h(x)≥0.5时令y=1，h(x)<0.5时令y=0，即可实现二分类。

```
LR_model = LR(
    penalty='l2',	# 惩罚项
    dual=False,		# 对偶(或原始)方法
    tol=0.0001,		# 停止求解的标准
    C=1.0,		# 正则化系数λ的倒数
    fit_intercept=True,	# 存在截距或偏差
    intercept_scaling=1,
    class_weight={0: 0.2, 1: 0.8},	# 分类模型中各类型的权重
    random_state=None,	# 随机种子
    solver='lbfgs',	# 优化算法选择参数
    max_iter=10,	# 算法收敛最大迭代次数
    multi_class='auto',	# 分类方式选择参数
    verbose=0,		# 日志冗长度
    warm_start=False,	# 热启动参数
    n_jobs=None,	# 并行数
    l1_ratio=None
)
```

*逻辑回归模型相关的Python脚本在“LR_train.py”中。*

#### 支持向量机

**支持向量机**（Support Vector Machine，SVM）是一类按监督学习方式对数据进行二元分类的广义线性分类器，对于小样本、复杂模型的学习表现出较好的效果。

支持向量机通过最大边距超平面实现类别的划分，即将上述特征视为高维空间上的点，并求解与两类点的边距最大的超平面，设为
$wx + b = 0$
。称同一类别点中到超平面距离最近的点为支持向量，记为
$z_{0}$
，则需要使支持向量到超平面的距离尽可能远。首先，任意一点x到超平面的距离为

$$
d = \frac{\left| {wx + b} \right|}{\left\| w \right\|}
$$

其中
$\left\| w \right\|$
选取w的2-范数，即
$\left\| w \right\| = \sqrt{\sum\limits_{i}w_{i}^{2}}$
。又由支持向量的定义，有

$$
\frac{\left| {wx + b} \right|}{\left\| w \right\|} \geq \frac{\left| {wz_{0} + b} \right|}{\left\| w \right\|} = d_{0}
$$

化简可得

$$
\left| \frac{wx + b}{\left\| w \right\| d_{0}} \right| \geq 1
$$

为便于进一步推导与优化，由
$\left\| w \right\| d_{0}$
为正数，可令其为1，则有

$$
\left| {wx + b} \right| \geq 1
$$

又因为要想使$d_{0}$尽可能大，应使
$\frac{1}{\left\| w \right\|}$
尽可能大，由此得出支持向量机模型

$$
\max\limits_{}\frac{1}{\left\| w \right\|} \quad s.t.\left| {wx + b} \right| \geq 1
$$

```
SVM_model = svm.SVC(
    C=1.0,		# 惩罚参数
    kernel='rbf',	# 核函数
    degree=3,		# 多项式'poly'维度
    gamma='auto',
    coef0=0.0,		# 核函数常数项
    shrinking=True,	# 采用shrinking heuristic方法
    probability=True,	# 采用概率估计
    tol=0.001,		# 停止训练的误差值
    cache_size=200,	# 核函数cache缓存
    class_weight=None,	# 类别权重
    verbose=False,	# 允许冗余输出
    max_iter=-1,	# 最大迭代次数无约束
    decision_function_shape='ovo',  # 多分类策略
    random_state=None,	# 随机种子
)
```

*支持向量机模型相关的Python脚本在“SVM_train.py”中。*

#### XGBoost

**XGBoost**（eXtreme Gradient Boosting，XGB）是梯度提升决策树（Gradient Boosting Decision Tree，GBDT）的一种，由集成的CART回归树构成，在很多情景下都表现出了出色的效率与较高的预测准确度。

XGBoost采用前向分布算法，学习包含K棵树的加法模型

$$
{\hat{y}}_{i} = {\sum\limits_{k = 1}^{K}{f_{k}\left( x_{i} \right)}}, \quad f \in F
$$

其中
$f_{k}$
为第k棵回归树模型；F对应回归树组成的函数空间。其目标函数定义为

$$
Obj(\Theta) = {\sum\limits_{i = 1}^{N}{l\left( {y_{i},{\hat{y}}_{i}} \right)}} + {\sum\limits_{j = 1}^{t}{\Omega\left( f_{j} \right)}}, \quad f_{j} \in F
$$

其中l为损失函数；Ω为正则化函数，与模型的复杂程度相关。正则项的加入能够有效防止模型过度拟合。

```
XGB_model = XGBClassifier(
    max_depth=6,	# 树的深度
    learning_rate=0.1,	# 学习率
    n_estimators=100,	# 迭代次数(决策树个数)
    silent=False,	# 是否输出中间过程
    objective='binary:logitraw',  # 目标函数
    booster='gbtree',	# 基分类器
    nthread=-1,		# 使用全部CPU进行并行运算
    gamma=1,		# 惩罚项系数
    min_child_weight=1,	# 最小叶子节点样本权重和
    max_delta_step=0,	# 对权重改变的最大步长无约束
    subsample=1,	# 每棵树训练集占全部训练集的比例
    colsample_bytree=1,	# 每棵树特征占全部特征的比例
    colsample_bylevel=1,
    eta=0.1,
    reg_alpha=0,	# L1正则化系数
    reg_lambda=1,	# L2正则化系数
    scale_pos_weight=0.5,  # 正样本的权重
    base_score=0.5,
    random_state=0,
    seed=None,		# 随机种子
    missing=None,
    use_label_encoder=False
)
```

*XGBoost模型相关的Python脚本在“XGB_train.py”中。*

### UEBA方法

UEBA是**用户和实体行为分析技术**（user and entity behavior analytics）的简称。它重点关注用户和实体的异常行为，能够基于海量数据对内部威胁进行预测，对异常的用户或实体行为进行判定，以尽早地排除风险，为安全智能分析提供可靠的依据。

UEBA分析方法大致可以分为以下几类：

- 有监督的机器学习  
已知正常行为和异常行为的集合被输入到系统中。该工具学习分析新行为并确定它是否类似于已知的正常或异常行为集。
- 贝叶斯网络  
可以结合监督的机器学习和规则来创建行为配置文件。
- 无监督学习  
系统学习正常行为，并能够检测和警告异常行为。它无法分辨出异常行为是好是坏，仅是它偏离了正常行为。
- 强化/半监督机器学习  
一种混合模型，其基础是无监督学习，并且将实际的警报解决方案被反馈到系统中，以允许对该模型进行精细调整并降低信噪比。
- 深度学习  
启用虚拟警报分类和调查。该系统训练代表安全警报及其分类结果的数据集，执行功能的自我识别，并能够预测新的安全警报集的分类结果。

经测试，经典监督学习模型在此问题上的表现并不理想，应深入研究UEBA分析方法，尝试其他模型。

一种合理的猜想是，用户的正常行为都比较类似，但异常行为的特征各异（亦或用户的异常行为都比较类似，但正常行为的特征各异），导致传统的监督学习难以区分正常和异常行为。

另一种猜想是，上面的模型仅适用于检测**离群点**（outlier detection），即存在于训练集中的异常点，而不适用于检测**奇异点**（novelty detection），即未在训练集中出现的新类型的样本。

由此，将已经完成预处理的原数据按照“risk_label（风险标识）”划分为正常行为集和异常行为集，引入下面模型。

#### 一类支持向量机

**一类支持向量机**（One Class Support Vector Machine，One Class SVM）是一类典型的单分类模型，常用于奇异点检测。

One Class SVM模型的训练集中应只包含一类行为。One Class SVM的定义方式有很多，比较常见的有以下两种。

在参考文献[1]中提出的One Class SVM方法（可简称为OCSVM）实质是将所有数据点与零点在特征空间F分离，并且最大化分离超平面到零点的距离。其优化目标与经典的SVM有所不同，要求

$$
{\min\limits_{w,\zeta_{i},\rho}{\frac{1}{2}\left\| w \right\|^{2}}} + \frac{1}{\nu n}{\sum\limits_{i = 1}^{n}\zeta_{i}} - \rho \quad s.t.\left( {w^{T}\phi\left( x_{i} \right)} \right) > \rho - \zeta_{i}, \quad i = 1,..,n
$$

其中
$\zeta_{i}$
为松弛变量且满足
$\zeta_{i} > 0$
，ν可以调整训练集中可信样本的比例。

在参考文献[2]中提出的One Class SVM方法（可简称为SVDD）实质是在特征空间中获得数据周围的球形边界，这个超球体的体积是最小化的，从而最小化异常点的影响。产生的超球体中心为a、半径为R，体积
$R^{2}$
被最小化，中心a是支持向量的线性组合。与经典的SVM方法相似，要求每个数据点
$x_{i}$
到中心的距离严格小于R，但同时构造一个惩罚系数为C的松弛变量
$\zeta_{i}$
满足
$\zeta_{i} > 0$
，优化问题为

$$
{\min\limits_{R,a}R^{2}} + C{\sum\limits_{i = 1}^{n}\zeta_{i}} \quad s.t.\left\| {x_{i} - a} \right\|^{2} \leq R^{2} + \zeta_{i}, \quad i = 1,..,n
$$

```
OneClassSVM_model = OneClassSVM(
    kernel='rbf',	# 核函数
    degree=3,		# 多项式维度
    gamma='auto',
    coef0=0.0,		# 核函数常数项
    shrinking=True,	# 采用shrinking heuristic方法
    tol=0.001,		# 停止训练的误差值
    cache_size=200,	# 核函数cache缓存
    verbose=False,	# 允许冗余输出
    max_iter=-1,	# 最大迭代次数无约束
)
```

*一类支持向量机模型相关的Python脚本在“OneClassSVM.py”中。*

#### 局部异常因子

**局部异常因子**（Local Outlier Factor，LOF）是一种适用于高维数据集的异常值检测方法，既可以检测奇异点，也可以检测离群点。因此可以看出，奇异点检测和离群点检测应分别属于有监督学习和无监督学习。

LOF的基本思想是，以一个数据点周围数据点所处位置的平均密度比上该数据点所处位置的密度来反映该数据点的异常程度。

在参考文献[3]中提出了一种方法，为此需定义k-距离（k-distance）、k-距离邻域（k-distance neighborhood）、可达距离（reachability distance）和局部可达密度（(local reachability density）等概念，最终通过局部离群因子（local outlier factor）来反映某个数据点的异常程度。具体定义参考文献中已经清晰地给出，后续也有多种优化方法被提出，这里就不再赘述。

```
LOF_model = LocalOutlierFactor(
    n_neighbors=20,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    contamination=0.5,
    novelty=False,	# 有监督学习
    n_jobs=None
)
```

*局部异常因子模型相关的Python脚本在“LocalOutlierFactor.py”中。*

#### 孤立森林

**孤立森林**（Isolation Forest）是一种无监督的异常检测方法，多适用于连续数据，由周志华教授等人于2008年在第八届IEEE数据挖掘国际会议上提出，之后凭借其线性的时间复杂度与优秀的准确率被广泛应用于各种结构化数据的异常检测。与其它算法不同的是，它的目标不再是描述正常的数据点，而是要孤立异常点，即分布稀疏且离密度高的群体较远的点。

参考文献[4]中提出了孤立森林的具体构建方法。孤立森林由孤立树（isolation tree，iTree）组成。构建iTree时，每次随机选取特征，并对节点进行二叉的随机划分。由此，被孤立的异常点会更早地被划分成单独的节点，而聚集的簇则需要更多次的切割，直至数据不可再划分或达到指定的最大深度。

```
IsolationForest_model = IsolationForest(
    n_estimators=100,	# 随机树数量
    max_samples='auto',	# 子采样的大小
    contamination=0.5,	# 异常数据占给定的数据集的比例
    max_features=1.0,	# 每棵树iTree的属性数量
    bootstrap=False,	# 执行不放回的采样
    n_jobs=-1,		# 并行运行的作业数量
    random_state=None,	# 随机数生成器
    verbose=0,		# 打印日志的详细程度
    warm_start=False	# fit时不重用上一次调用的结果
)
```

*孤立森林模型相关的Python脚本在“IsolationForest.py”中。*

### 集成学习

**集成学习**（Ensemble Learning）是指通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统、基于委员会的学习等。

在集成学习的一般结构中，先产生一组“个体学习器”，再采用某种策略将它们结合起来。集成中只包含同种类型的个体学习器，称为同质，其中的个体学习器亦称为“基学习器”，相应的算法称为“基学习算法”；集成中包含不同类型的个体学习器，称为“异质”，其中的个体学习器称为“组建学习器”。从上面的描述中可以发现，要想获得好的集成学习模型，个体学习器应“好而不同”，即单个个体学习器要有一定的“准确性”，同时多个个体学习器之间要保持多样性。

在本实验中，上述“模型架构与算法原理”部分阐述的所有模型均可以成为个体学习器。通过个体学习器的预测结果可以分析出，不同的个体学习器（尤其是UEBA方法）对正常行为和异常行为的预测表现出不同的结果。因此可以利用这点，分别为正常行为和异常行为建立模型，即为每个个体学习器设定合理的权值，令其对正常行为/异常行为进行预测，加权求和后，综合得出结果，即

$$Score = {\sum\limits_{i = 1}^{n}{S_{i} \times P_{i}}}$$

其中，Score为最终预测结果，
$S_{i}$
为第i个学习器的预测结果，
$P_{i}$
为第i个学习器的权值。

## 参赛结果

最终提交结果后，最好成绩和排名如下图所示。（该赛题A榜/B榜第1名的最好成绩分别为0.53471494/0.53765515。）

评测阶段|最好成绩|排名
:---:|:---:|:---:
A榜|0.51027393|226
B榜|0.50779870|262

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/certificate.png)

根据评测标准，本赛题使用ROC曲线下面积AUC（Area Under Curve）作为评价指标。AUC值越大，预测越准确。具体公式为

$$AUC = \frac{\sum_{i \in positiveClass}{{rank}_{i} - \frac{M(1 + M)}{2}}}{M \times N}$$

其中，M为正样本，N为负样本。当正样本预测为正样本的概率值大于负样本预测为正样本的概率值记为1，并累加计数，然后除以M×N样本对，即AUC的值。

从第1名的最好成绩也极接近0.5可以看出，该赛题的训练集和测试集之间的相似性并没有想像中的大。若从训练集中划分出部分样本作为测试集，其它样本以同样方式建立模型，得出预测结果的AUC也远远大于提交结果后得到的成绩，这也进一步验证了上面的观点。

## 参考文献

[1] Bernhard H Schölkopf, Robert C Williamson, Alexander Smola, John C Shawe-Taylor, John C Platt. Support vector method for novelty detection[C]. NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems, 1999.  
[2] David M.J. Tax, Robert P.W. Duin. Support Vector Data Description[J]. Machine Learning, 2004, 54: 45-66.  
[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond Tak Yan Ng, Jörg Sander. LOF: identifying density-based local outliers[C]. Proc. ACM SIGMOD 2000 Int. Conf. On Management of Data, 2000.  
[4] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest[C]. IEEE International Conference on Data Mining, 2008.  
[5] 朱佳俊, 陈功, 施勇, 薛质. 基于用户画像的异常行为检测[J]. 通信技术, 2017, 50(10): 2310-2315.  
[6] 崔景洋, 陈振国, 田立勤, 张光华. 基于机器学习的用户与实体行为分析技术综述[J/OL]. 计算机工程. https://doi.org/10.19678/j.issn.1000-3428.0062623.  
[7] 爱丽丝·郑, 阿曼达·卡萨丽. 精通特征工程[M]. 北京: 人民邮电出版社, 2019.
