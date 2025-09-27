# Certification Risk Prediction

<div align="right">
	[<a href="README.md">中文</a> | English(Current)</a>]
</div>

## Related Project

[Abnormal Accounts Recognition](https://github.com/baoyunfan0101/AbnormalAccountsRecognition)

## File Description

datasets // datasets (training set, test set)  
feature engineering // feature engineering  
models // evaluation models

## Test Environment

Python3.8

## Task Description

The project comes from the competition task [System Authentication Risk Prediction](https://www.datafountain.cn/competitions/537).

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/problem.png)

In this task, participating teams will build a user authentication behavior feature model and a risk anomaly evaluation model based on user authentication behavior data and risk anomaly labels, and use the risk evaluation model to determine whether the current user authentication behavior is risky.

* Build behavioral baselines from user authentication data;
* Use supervised learning models based on authentication behavior features to build a risk anomaly evaluation model and determine whether the current authentication behavior is risky.

## Feature Engineering

### Data Preprocessing

The raw data contains 18 features and 1 label. Many features are discrete information without learning value, so we describe below the preprocessing method applied to each feature.

---

* `session_id` (authentication ID)  
Check for duplicate authentication IDs; remove duplicate rows, then drop this feature.
* `op_date` (authentication time)  
First find records with the same user name and IP address. For each record, from the records where both user name and IP are the same, find the most recent record prior to the current one, compute the authentication time difference, insert it into the dataset as a new feature `op_timedelta` (authentication time difference), and drop the original authentication time.
* `user_name` (user name)  
No learning value; drop.
* `action` (operation type)  
Has two possible values, “login” and “sso”; replace them with 0 and 1 respectively.
* `auth_type` (first authentication method)  
Has five possible values: “pwd”, “sms”, “otp”, “qr”, and “(none)”. Apply one-hot encoding to convert into five boolean discrete features. In particular, “(none)” has a strong impact on the prediction; and because the first four features might be dropped in subsequent steps, keep “(none)” as a separate feature.
* `ip` (IP address)  
No learning value; drop.
* `ip_location_type_keyword` (IP type)  
Has four possible values: “home broadband”, “proxy IP”, “intranet”, and “public broadband”. Apply one-hot encoding to convert into four boolean discrete features.
* `ip_risk_level` (IP threat level)  
Has three possible values: “level 1”, “level 2”, and “level 3”. Because of the ordinal relationship, replace them with 1, 2, and 3 respectively.
* `location` (location)  
Limited learning value; drop.
* `client_type` (client type)  
Has two possible values: “app” and “web”; replace with 0 and 1 respectively.
* `browser_source` (browser source)  
Has two possible values: “desktop” and “mobile”; replace with 0 and 1 respectively.
* `device_model` (device model)  
Limited learning value; drop.
* `os_type` (operating system type)  
Has two possible values: “windows” and “macOS”; replace with 0 and 1 respectively.
* `os_version` (operating system version)  
Limited learning value; drop.
* `browser_type` (browser type)  
Has five possible values: “edge”, “chrome”, “firefox”, “ie”, and “safari”. Apply one-hot encoding to convert into five boolean discrete features.
* `browser_version` (browser version)  
Limited learning value; drop.
* `bus_system_code` (application system code)  
Has seven possible values: “attendance”, “coremail”, “crm”, “oa”, “order-mgnt”, “reimbursement”, and “salary”. Apply one-hot encoding to convert into seven boolean discrete features.
* `op_target` (application system category)  
Has four possible values: “sales”, “finance”, “management”, and “hr”. Apply one-hot encoding to convert into four boolean discrete features.

---

Through the above preprocessing, the 18 original features are encoded into 31 new features. The training set comes with the label `risk_label` (risk indicator).

Because the feature scales are inconsistent and there are outliers outside the normal range, data standardization is required. Here we perform z-score standardization using the mean and standard deviation of the original data to meet the needs of the following model training. The formula is

$$
{X'}_{i} = \frac{X_{i} - {\overset{-}{X}}_{i}}{S}
$$

where
${X'}_{i}$
is the standardized feature;
$X_{i}$
is the original feature;
${\overset{-}{X}}_{i}$
is the mean of the original feature; and \(S\) is the standard deviation of the original feature, computed as
\(\sqrt{\frac{\sum\limits_{i = 1}^{n}\left( {x_{i} - \overset{-}{x}} \right)^{2}}{n - 1}}\)
.

*The Python script for data preprocessing is in “preprocessing.py”.*

### Feature Derivation and Selection

Operations and transaction information of the same account are naturally the focus of an account feature model, and their timing information (corresponding to attribute `tm_diff`) is the most critical part of modeling. To this end, inspired by the RFM analytics method, we derive features from the relevant timing information.

RFM stands for Recency (time since the most recent transaction), Frequency (transaction frequency), and Monetary (transaction amount). Following this idea, we extract four features from account operation information: latest operation time `op_recent_tm`, operation frequency `op_frequency`, average operation interval `op_interval`, and minimum operation interval `op_min_interval`; and five features from account transaction information: latest transaction time `trans_recent_tm`, transaction frequency `trans_frequency`, transaction amount `trans_amount`, average transaction interval `trans_interval`, and minimum transaction interval `trans_min_interval`.

There is a specific reason to keep both the average interval and the minimum interval. On the one hand, from a professional standpoint, the minimum operation/transaction interval is an important criterion for judging whether an account is handled manually, and thus has special value for identifying abnormal accounts; on the other hand, the average interval only relates to the earliest and latest operations/transactions of an account, while adding the minimum interval enables more effective use of the data and more completely reflects the concept of Frequency in RFM.

During feature selection, in addition to removing attributes with excessive missing values mentioned in the “Data Preprocessing” section above, we performed further screening based on the results of the following feature analysis, which will be described in detail below.

*Feature derivation for both the training and test sets is also in `preprocessing_train.py` and `preprocessing_test.py` and is performed alongside preprocessing. The Python scripts for feature selection in the training and test sets are `screening_train.py` and `screening_test.py`, respectively.*

### Feature Analysis

*The Python script for feature analysis is in `iv.py`.*

#### Feature Importance Evaluation

**WOE** (Weight of Evidence) is an encoding of an original independent variable. After grouping/discretizing with respect to an evaluation criterion, it is computed by

$$
{WOE}_{i} = ln\left( \frac{{py}_{i}}{{pn}_{i}} \right) = ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

where
${WOE}_{i}$
is the WOE of the \(i\)-th group;
${py}_{i}$
is the proportion of responding customers (i.e., risky accounts in this task) in the \(i\)-th group among all responding samples;
${pn}_{i}$
is the proportion of non-responding customers in the \(i\)-th group among all non-responding samples;
$y_{i}$
is the number of responding customers in the \(i\)-th group;
$y_{T}$
is the number of non-responding customers in the \(i\)-th group;
$n_{i}$
is the number of responding customers in all samples; and
$n_{T}$
is the number of non-responding customers in all samples.

**IV** (Information Value) takes into account both the WOE of each group and its share in the overall sample. It can be seen as a weighted sum of WOE and, in this task, reflects the contribution of a given feature to account risk. The specific formula for the IV of one group is

$$
{IV}_{i} = \left( {py}_{i} - {pn}_{i} \right) \times {WOE}_{i} = \left( \frac{y_{i}}{y_{T}} - \frac{n_{i}}{n_{T}} \right) \times ln\left( \frac{\frac{y_{i}}{y_{T}}}{\frac{n_{i}}{n_{T}}} \right)
$$

where
${IV}_{i}$
is the IV of the \(i\)-th group. The IV of a feature is

$$
IV = \sum\limits_{i = 1}^{n}{IV}_{i}
$$

where \(n\) is the number of groups.

Considering the 31 features obtained in the preprocessing step above, except for boolean discrete features that only take values 0 and 1, we perform chi-square binning using each of the other features in turn, splitting all data into two groups (in particular, for the feature `op_timedelta` we split into five groups), and then compute their IVs. The results are as follows.

Feature|IV(*1e-4)
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

Observing the IVs of each feature, it is easy to find that the following five features have low importance relative to `op_timedelta` (authentication time difference): `client_type` (client type), `browser_source` (browser source), `ip_location_type_keyword_2` (IP type: proxy IP), `ip_location_type_keyword_4` (IP type: public broadband), and `op_target_1` (application category: sales). Therefore these five features may be dropped in subsequent modeling as appropriate.

#### Feature Correlation Analysis

We compute the correlation matrix for the above evaluation indicators and plot a heatmap, as shown below.

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/correlation.png)

From the heatmap, excluding the five features to be dropped as mentioned above, the correlations among the remaining features are generally within a reasonable range; there are no overly correlated features. This indirectly supports the reasonableness of feature extraction during preprocessing.

## Model Training and Optimization

### Classical Supervised Learning Models

In the risk behavior detection model, let the dependent variable `risk_label` take values \(y \in \{0,1\}\); this is a binary classification problem.

#### Logistic Regression

**Logistic Regression (LR)** is a generalized linear regression model commonly used for binary classification problems.

If under independent variables \(x=X\) the probability that \(y=1\) is \(p\), written as
\(p = P\left( y = 1 \middle| X \right)\),
then the probability of \(y=0\) is
\(1 - p\).
Let the odds be the ratio
\(\frac{p}{1 - p}\),
and taking the natural logarithm gives the Sigmoid function

$$
Sigmoid(p) = ln\left( \frac{p}{1 - p} \right)
$$

Let
\(Sigmoid(p) = z\),
then

$$
p = \frac{1}{1 + e^{- z}}
$$

Let the feature vector be \(X\) and the coefficient vector be \(\beta\). Substituting \(z\) above yields the regression model

$$
h(x) = \frac{1}{1 + e^{- X\beta^{T}}}
$$

Here \(h(x) \in [0,1]\), which can serve as the predicted risk label required in this task. Let \(y=1\) when \(h(x)\ge 0.5\) and \(y=0\) when \(h(x)<0.5\) to implement binary classification.

```python
LR_model = LR(
    penalty='l2',      # penalty term
    dual=False,        # use dual (or primal) formulation
    tol=0.0001,        # stopping tolerance
    C=1.0,             # inverse of regularization strength λ
    fit_intercept=True,# include intercept
    intercept_scaling=1,
    class_weight={0: 0.2, 1: 0.8}, # class weights
    random_state=None, # random seed
    solver='lbfgs',    # optimization algorithm
    max_iter=10,       # max iterations
    multi_class='auto',# multiclass setting
    verbose=0,         # verbosity
    warm_start=False,  # warm start
    n_jobs=None,       # number of parallel jobs
    l1_ratio=None
)
```

*The Python script for the logistic regression model is in `LR_train.py`.*

#### Support Vector Machine

A **Support Vector Machine (SVM)** is a family of generalized linear classifiers that performs binary classification via supervised learning and often works well for small-sample and complex models.

An SVM separates classes using a maximum-margin hyperplane. Viewing the features as points in a high-dimensional space, we seek the hyperplane with the largest margin between the two classes, denoted by
\(wx + b = 0\).
The closest points of each class to the hyperplane are the support vectors, denoted by
\(z_{0}\).
We want the distance from a support vector to the hyperplane to be as large as possible. For any point \(x\), the distance to the hyperplane is

$$
d = \frac{\left| {wx + b} \right|}{\left\| w \right\|}
$$

where
\(\left\| w \right\|\)
is the 2-norm of \(w\),
\(\left\| w \right\| = \sqrt{\sum\limits_{i}w_{i}^{2}}\).
By the definition of a support vector,

$$
\frac{\left| {wx + b} \right|}{\left\| w \right\|} \geq \frac{\left| {wz_{0} + b} \right|}{\left\| w \right\|} = d_{0}
$$

which simplifies to

$$
\left| \frac{wx + b}{\left\| w \right\| d_{0}} \right| \geq 1
$$

For convenience, since
\(\left\| w \right\| d_{0} > 0\),
let it be 1, giving

$$
\left| {wx + b} \right| \geq 1
$$

To maximize \(d_0\), we should maximize
\(\frac{1}{\left\| w \right\|}\).
Thus the SVM model is

$$
\max\limits_{}\frac{1}{\left\| w \right\|} \quad s.t.\left| {wx + b} \right| \geq 1
$$

```python
SVM_model = svm.SVC(
    C=1.0,            # penalty parameter
    kernel='rbf',     # kernel
    degree=3,         # degree for 'poly'
    gamma='auto',
    coef0=0.0,        # kernel constant term
    shrinking=True,   # use shrinking heuristic
    probability=True, # enable probability estimates
    tol=0.001,        # tolerance for stopping
    cache_size=200,   # kernel cache (MB)
    class_weight=None,# class weights
    verbose=False,    # verbosity
    max_iter=-1,      # no max iteration limit
    decision_function_shape='ovo',  # multi-class strategy
    random_state=None,# random seed
)
```

*The Python script for the support vector machine model is in `SVM_train.py`.*

#### XGBoost

**XGBoost** (eXtreme Gradient Boosting, XGB) is a form of Gradient Boosting Decision Trees (GBDT), consisting of an ensemble of CART regression trees. It often achieves high efficiency and predictive accuracy in many scenarios.

XGBoost uses a forward stagewise additive model with \(K\) trees:

$$
{\hat{y}}_{i} = {\sum\limits_{k = 1}^{K}{f_{k}\left( x_{i} \right)}}, \quad f \in F
$$

where
\(f_{k}\)
is the \(k\)-th regression tree; \(F\) is the function space of trees. The objective is

$$
Obj(\Theta) = {\sum\limits_{i = 1}^{N}{l\left( {y_{i},{\hat{y}}_{i}} \right)}} + {\sum\limits_{j = 1}^{t}{\Omega\left( f_{j} \right)}}, \quad f_{j} \in F
$$

where \(l\) is the loss and \(\Omega\) is the regularization term related to model complexity. Regularization helps prevent overfitting.

```python
XGB_model = XGBClassifier(
    max_depth=6,      # tree depth
    learning_rate=0.1,# learning rate
    n_estimators=100, # number of boosting rounds
    silent=False,     # output intermediate logs
    objective='binary:logitraw',  # objective
    booster='gbtree', # base learner
    nthread=-1,       # use all CPUs
    gamma=1,          # penalty term
    min_child_weight=1,# min sum of instance weight in a leaf
    max_delta_step=0, # no max delta step constraint
    subsample=1,      # subsample ratio of training instances
    colsample_bytree=1, # subsample ratio of columns per tree
    colsample_bylevel=1,
    eta=0.1,
    reg_alpha=0,      # L1 regularization
    reg_lambda=1,     # L2 regularization
    scale_pos_weight=0.5,  # positive class weight
    base_score=0.5,
    random_state=0,
    seed=None,        # random seed
    missing=None,
    use_label_encoder=False
)
```

*The Python script for the XGBoost model is in `XGB_train.py`.*

### UEBA Methods

UEBA stands for **User and Entity Behavior Analytics**. It focuses on anomalous user and entity behaviors, enabling the prediction of insider threats based on massive data and the identification of abnormal user or entity behavior, so that risks can be mitigated early and reliable evidence provided for security analytics.

UEBA analysis methods can be roughly divided into the following categories:

* **Supervised machine learning**  
A set of known normal and abnormal behaviors is input to the system. The tool learns to analyze new behaviors and determines whether they resemble the known sets of normal or abnormal behaviors.
* **Bayesian networks**  
Can combine supervised machine learning and rules to create behavioral profiles.
* **Unsupervised learning**  
The system learns normal behavior and can detect and alert on abnormal behavior. It cannot tell whether an abnormal behavior is good or bad; only that it deviates from normal behavior.
* **Reinforcement/semi-supervised machine learning**  
A hybrid model based on unsupervised learning that feeds back actual alert resolutions into the system to fine-tune the model and reduce noise.
* **Deep learning**  
Enables virtual alert classification and investigation. The system is trained on datasets representing security alerts and their classification outcomes, performs feature self-identification, and can predict classification results for new sets of security alerts.

In our tests, classical supervised learning models did not perform well on this problem. We should study UEBA methods more deeply and try other models.

One reasonable hypothesis is that normal user behaviors are relatively similar while anomalous behaviors vary widely (or the inverse), making it difficult for traditional supervised learning to distinguish normal from abnormal behaviors.

Another hypothesis is that the above models are only suitable for **outlier detection** (detecting anomalies that exist in the training set) but not for **novelty detection** (detecting new types of samples not seen in the training set).

Therefore, we split the preprocessed data into normal and abnormal behavior sets according to `risk_label` and introduce the following models.

#### One-Class Support Vector Machine

A **One-Class Support Vector Machine (One-Class SVM)** is a typical single-class model, commonly used for novelty detection.

The training set for a One-Class SVM should contain only one class of behavior. There are several formulations; two common ones are as follows.

The One-Class SVM proposed in reference [1] (abbreviated **OCSVM**) essentially separates all data points from the origin in the feature space \(F\) with a hyperplane and maximizes the distance of this hyperplane from the origin. Its optimization objective differs from classical SVM and requires

$$
{\min\limits_{w,\zeta_{i},\rho}{\frac{1}{2}\left\| w \right\|^{2}}} + \frac{1}{\nu n}{\sum\limits_{i = 1}^{n}\zeta_{i}} - \rho \quad s.t.\left( {w^{T}\phi\left( x_{i} \right)} \right) > \rho - \zeta_{i}, \quad i = 1,..,n
$$

where
\(\zeta_{i}\)
are slack variables with
\(\zeta_{i} > 0\),
and \(\nu\) controls the proportion of trusted samples in the training set.

The One-Class SVM proposed in reference [2] (abbreviated **SVDD**) essentially obtains a spherical boundary around the data in the feature space; the volume of this hypersphere is minimized to reduce the influence of outliers. The resulting hypersphere has center \(a\) and radius \(R\). The volume
\(R^{2}\)
is minimized, and the center \(a\) is a linear combination of support vectors. Similar to classical SVM, each data point
\(x_{i}\)
is required to be strictly within radius \(R\), while introducing slack variables
\(\zeta_{i}\)
with penalty coefficient \(C\) such that
\(\zeta_{i} > 0\). The optimization problem is

$$
{\min\limits_{R,a}R^{2}} + C{\sum\limits_{i = 1}^{n}\zeta_{i}} \quad s.t.\left\| {x_{i} - a} \right\|^{2} \leq R^{2} + \zeta_{i}, \quad i = 1,..,n
$$

```python
OneClassSVM_model = OneClassSVM(
    kernel='rbf',     # kernel
    degree=3,         # degree for polynomial
    gamma='auto',
    coef0=0.0,        # kernel constant term
    shrinking=True,   # use shrinking heuristic
    tol=0.001,        # tolerance for stopping
    cache_size=200,   # kernel cache (MB)
    verbose=False,    # verbosity
    max_iter=-1,      # no max iteration limit
)
```

*The Python script for the One-Class SVM model is in `OneClassSVM.py`.*

#### Local Outlier Factor

**Local Outlier Factor (LOF)** is an anomaly detection method suited to high-dimensional datasets; it can detect both novelties and outliers. Thus, novelty detection and outlier detection should be considered as belonging to supervised and unsupervised learning respectively.

The basic idea of LOF is to reflect the abnormality of a data point by comparing the average density of its surrounding points with the density at the point itself.

Reference [3] proposes a method that defines the notions of k-distance, k-distance neighborhood, reachability distance, and local reachability density, and ultimately uses the local outlier factor to quantify the degree of abnormality of a point. The specific definitions are clearly provided in the reference, and many optimizations have been proposed since; we do not repeat them here.

```python
LOF_model = LocalOutlierFactor(
    n_neighbors=20,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    contamination=0.5,
    novelty=False,   # supervised learning
    n_jobs=None
)
```

*The Python script for the Local Outlier Factor model is in `LocalOutlierFactor.py`.*

#### Isolation Forest

**Isolation Forest** is an unsupervised anomaly detection method, typically for continuous data. Proposed by Zhou Zhihua et al. in 2008 at the 8th IEEE International Conference on Data Mining, it has been widely used in anomaly detection for structured data due to its linear time complexity and strong accuracy. Unlike other algorithms, it does not model normal data; instead, it isolates anomalies—points that lie in sparse regions and far from dense clusters.

Reference [4] presents the construction of Isolation Forest. An Isolation Forest consists of isolation trees (iTrees). When building an iTree, at each step randomly select a feature and split the node with a random threshold. As a result, isolated anomalies are split into single nodes earlier, whereas clustered points require more cuts until the data cannot be further split or a maximum depth is reached.

```python
IsolationForest_model = IsolationForest(
    n_estimators=100,   # number of random trees
    max_samples='auto', # subsample size
    contamination=0.5,  # proportion of anomalies in the dataset
    max_features=1.0,   # number of features for each iTree
    bootstrap=False,    # sampling without replacement
    n_jobs=-1,          # number of parallel jobs
    random_state=None,  # random state
    verbose=0,          # verbosity
    warm_start=False    # do not reuse previous fit
)
```

*The Python script for the Isolation Forest model is in `IsolationForest.py`.*

### Ensemble Learning

**Ensemble Learning** refers to building and combining multiple learners to complete a task; it is also known as multiple classifier systems or committee-based learning.

In a typical ensemble, we first produce a set of “individual learners” and then combine them using some strategy. If the ensemble contains only one type of individual learner, it is homogeneous, and the individual learners are “base learners” trained by a “base learning algorithm”. If it contains different types, it is heterogeneous, and the individual learners are “component learners”. To achieve a strong ensemble, individual learners should be “good but different”: each single learner should have certain accuracy, and multiple learners should maintain diversity.

In our experiments, all models described in the “Model Architecture and Algorithm Principles” section above can be used as individual learners. From their predictions (especially from UEBA methods), we observe differences in how they predict normal vs. abnormal behaviors. We can exploit this by building models separately for normal and abnormal behaviors: assign a reasonable weight to each learner, obtain its prediction for normal/abnormal behavior, and take a weighted sum to produce the final result:

$$Score = {\sum\limits_{i = 1}^{n}{S_{i} \times P_{i}}}$$

where \(Score\) is the final prediction,
\(S_{i}\) is the \(i\)-th learner’s prediction, and
\(P_{i}\) is the \(i\)-th learner’s weight.

## Competition Results

After submission, the best scores and rankings are shown below. (For this problem, the top scores on the A/B leaderboards are 0.53471494/0.53765515, respectively.)

Phase|Best Score|Rank
:---:|:---:|:---:
A leaderboard|0.51027393|226
B leaderboard|0.50779870|262

![image](https://github.com/baoyunfan0101/CertificationRiskPrediction/blob/main/static/certificate.png)

According to the evaluation standard, this task uses the area under the ROC curve (AUC) as the metric. The larger the AUC, the more accurate the prediction. The specific formula is

$$AUC = \frac{\sum_{i \in positiveClass}{{rank}_{i} - \frac{M(1 + M)}{2}}}{M \times N}$$

where \(M\) is the number of positive samples and \(N\) the number of negative samples. When the predicted probability that a positive sample is positive is greater than that of a negative sample being positive, count 1 and accumulate; divide by \(M \times N\) sample pairs to obtain the AUC.

From the top score being close to 0.5, we can see that the similarity between the training and test sets in this competition is not as high as one might expect. If we split part of the training samples as a test set and train on the remaining samples in the same way, the AUC on the holdout set is much higher than the submitted score, which further corroborates this view.

## References

[1] Bernhard H Schölkopf, Robert C Williamson, Alexander Smola, John C Shawe-Taylor, John C Platt. Support vector method for novelty detection[C]. NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems, 1999.  
[2] David M.J. Tax, Robert P.W. Duin. Support Vector Data Description[J]. Machine Learning, 2004, 54: 45-66.  
[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond Tak Yan Ng, Jörg Sander. LOF: identifying density-based local outliers[C]. Proc. ACM SIGMOD 2000 Int. Conf. on Management of Data, 2000.  
[4] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. Isolation Forest[C]. IEEE International Conference on Data Mining, 2008.  
[5] Jiajun Zhu, Gong Chen, Yong Shi, Zhi Xue. Abnormal Behavior Detection Based on User Profiling[J]. Communications Technology, 2017, 50(10): 2310-2315.  
[6] Jingyang Cui, Zhenguo Chen, Liqin Tian, Guanghua Zhang. A Survey of User and Entity Behavior Analytics Based on Machine Learning[J/OL]. Computer Engineering. https://doi.org/10.19678/j.issn.1000-3428.0062623.  
[7] Alice Zheng, Amanda Casari. Mastering Feature Engineering[M]. Beijing: Posts & Telecom Press, 2019.