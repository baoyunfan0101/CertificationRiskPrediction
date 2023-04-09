import pandas as pd
import numpy as np
from scipy.stats import chi2


# 计算卡方统计量
def cal_chi2(input_df, var_name, Y_name):
    input_df = input_df[[var_name, Y_name]]
    all_cnt = input_df[Y_name].count()
    all_0_cnt = input_df[input_df[Y_name] == 0].shape[0]
    all_1_cnt = input_df[input_df[Y_name] == 1].shape[0]
    expect_0_ratio = all_0_cnt * 1.0 / all_cnt
    expect_1_ratio = all_1_cnt * 1.0 / all_cnt
    # 对变量的每个值计算实际个数，期望个数，卡方统计量
    var_values = sorted(list(set(input_df[var_name])))
    actual_0_cnt = []
    actual_1_cnt = []
    actual_all_cnt = []
    expect_0_cnt = []
    expect_1_cnt = []
    chi2_value = []
    for value in var_values:
        actual_0 = input_df[(input_df[var_name] == value) & (input_df[Y_name] == 0)].shape[0]
        actual_1 = input_df[(input_df[var_name] == value) & (input_df[Y_name] == 1)].shape[0]
        actual_all = actual_0 + actual_1
        expect_0 = actual_all * expect_0_ratio
        expect_1 = actual_all * expect_1_ratio
        chi2_0 = (expect_0 - actual_0) ** 2 / (expect_0 if expect_0 != 0 else 1)
        chi2_1 = (expect_1 - actual_1) ** 2 / (expect_1 if expect_1 != 0 else 1)
        actual_0_cnt.append(actual_0)
        actual_1_cnt.append(actual_1)
        actual_all_cnt.append(actual_all)
        expect_0_cnt.append(expect_0)
        expect_1_cnt.append(expect_1)
        chi2_value.append(chi2_0 + chi2_1)

    chi2_result = pd.DataFrame({'actual_0': actual_0_cnt, 'actual_1': actual_1_cnt, 'expect_0': expect_0_cnt,
                                'expect_1': expect_1_cnt, 'chi2_value': chi2_value, var_name + '_start': var_values,
                                var_name + '_end': var_values},
                               columns=[var_name + '_start', var_name + '_end', 'actual_0', 'actual_1', 'expect_0',
                                        'expect_1', 'chi2_value'])

    return chi2_result, var_name


# 定义合并区间的方法
def merge_area(chi2_result, var_name, idx, merge_idx):
    # 按照idx和merge_idx执行合并
    chi2_result.loc[idx, 'actual_0'] = chi2_result.loc[idx, 'actual_0'] + chi2_result.loc[merge_idx, 'actual_0']
    chi2_result.loc[idx, 'actual_1'] = chi2_result.loc[idx, 'actual_1'] + chi2_result.loc[merge_idx, 'actual_1']
    chi2_result.loc[idx, 'expect_0'] = chi2_result.loc[idx, 'expect_0'] + chi2_result.loc[merge_idx, 'expect_0']
    chi2_result.loc[idx, 'expect_1'] = chi2_result.loc[idx, 'expect_1'] + chi2_result.loc[merge_idx, 'expect_1']
    chi2_0 = (chi2_result.loc[idx, 'expect_0'] - chi2_result.loc[idx, 'actual_0']) ** 2 / \
             (chi2_result.loc[idx, 'expect_0'] if chi2_result.loc[idx, 'expect_0'] != 0 else 1)
    chi2_1 = (chi2_result.loc[idx, 'expect_1'] - chi2_result.loc[idx, 'actual_1']) ** 2 / \
             (chi2_result.loc[idx, 'expect_1'] if chi2_result.loc[idx, 'expect_1'] != 0 else 1)
    chi2_result.loc[idx, 'chi2_value'] = chi2_0 + chi2_1
    # 调整每个区间的起始值
    if idx < merge_idx:
        chi2_result.loc[idx, var_name + '_end'] = chi2_result.loc[merge_idx, var_name + '_end']
    else:
        chi2_result.loc[idx, var_name + '_start'] = chi2_result.loc[merge_idx, var_name + '_start']
    chi2_result = chi2_result.drop([merge_idx])
    chi2_result = chi2_result.reset_index(drop=True)

    return chi2_result


# 自动进行分箱，使用最大区间限制
def chiMerge_maxInterval(chi2_result, var_name, max_interval=5):
    groups = chi2_result.shape[0]
    while groups > max_interval:
        min_idx = chi2_result[chi2_result['chi2_value'] == chi2_result['chi2_value'].min()].index.tolist()[0]
        if min_idx == 0:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)
        elif min_idx == groups - 1:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)
        else:
            if chi2_result.loc[min_idx - 1, 'chi2_value'] > chi2_result.loc[min_idx + 1, 'chi2_value']:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)
            else:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)
        groups = chi2_result.shape[0]

    return chi2_result


# 使用已经分箱的结果，计算woe和iv
def cal_woe(input_df):
    groups = input_df.shape[0]
    # 对于统计项为0的actual_0和actual_1赋值为1
    input_df.loc[input_df['actual_0'] == 0, 'actual_0'] = 1
    input_df.loc[input_df['actual_1'] == 0, 'actual_1'] = 1
    all_0 = input_df['actual_0'].sum()
    all_1 = input_df['actual_1'].sum()
    woe = []
    iv = 0
    for i in range(groups):
        py = input_df.loc[i, 'actual_1'] * 1.0 / all_1
        pn = input_df.loc[i, 'actual_0'] * 1.0 / all_0
        tmp = (py - pn) * np.log(py / pn)
        woe.append(tmp)
        iv += tmp

    return woe, iv


if __name__ == '__main__':
    # 导入数据
    df = pd.read_excel('train.xlsx', sheet_name='Sheet1')
    df = df.drop('Unnamed: 0', axis=1)

    # 计算IV
    print('IV(*10000):')
    for i in df.drop('risk_label', axis=1).columns.values.tolist():
        chi2_result, var_name = cal_chi2(df, i, 'risk_label')
        if (i == 'op_timedelta'):
            interval = 5
        else:
            interval = 2
        chi2_result = chiMerge_maxInterval(chi2_result, var_name, interval)
        # print(chi2_result)
        woe, iv = cal_woe(chi2_result)
        print(i, iv * 10000)

    print()

    # 计算相关系数
    correlation = df.drop('risk_label', axis=1).corr(method="spearman")
    print('correlation:')
    print(correlation)
    # 绘制热力图
    import matplotlib.pyplot as plt
    import seaborn as sns

    f, ax = plt.subplots()
    sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w', fmt='.2f', annot=False, square=True)
    plt.show()
