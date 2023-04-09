import pandas as pd

# train_dataset(训练集)
df1 = pd.read_csv('train_dataset.csv', sep='\t')

# session_id(认证ID):drop
for idx in range(0, len(df1)):
    i = df1.loc[idx, :]
    tmp = df1[df1.session_id == i['session_id']]
    if len(tmp) > 1:
        print('找到:', idx)
print('未找到')
df1 = df1.drop('session_id', axis=1)

# op_date(认证时间):(as follows)
t = []
for idx in range(0, len(df1)):
    i = df1.loc[idx, :]
    tmp = df1[(df1.user_name == i['user_name']) & (df1.ip == i['ip']) & (pd.to_datetime(
        df1.op_date) < pd.to_datetime(i['op_date']))]
    if len(tmp) == 0:
        t.append(100)
    else:
        min = (pd.to_datetime(i['op_date']) - (pd.to_datetime(tmp['op_date'])).max()).total_seconds() / 60
        min = 100 if min > 100 else min
        t.append(min)
df1['op_timedelta'] = t
df1 = df1.drop('op_date', axis=1)

# user_name(用户名):drop
df1 = df1.drop('user_name', axis=1)

# action(操作类型):replace
df1['action'] = df1['action'].replace({'login': 0, 'sso': 1})

# auth_type(首次认证方式):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []  # auth_type == None
for i in df1['auth_type']:
    if i == 'pwd':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t1.append(0)
    if i == 'sms':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t2.append(0)
    if i == 'otp':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t3.append(0)
    if i == 'qr':
        t4.append(1)
        t5.append(0)
    else:
        t4.append(0)
        t5.append(1)
df1 = df1.drop('auth_type', axis=1)
df1['auth_type_1'] = t1
df1['auth_type_2'] = t2
df1['auth_type_3'] = t3
df1['auth_type_4'] = t4
df1['auth_type_5'] = t5

# ip(IP地址):drop
df1 = df1.drop('ip', axis=1)

# ip_location_type_keyword(IP类型):ont-hot
t1 = []
t2 = []
t3 = []
t4 = []
for i in df1['ip_location_type_keyword']:
    if i == '家庭宽带':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t1.append(0)
    if i == '代理IP':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t2.append(0)
    if i == '内网':
        t3.append(1)
        t4.append(0)
        continue
    else:
        t3.append(0)
    if i == '公共宽带':
        t4.append(1)
    else:
        t4.append(0)
df1 = df1.drop('ip_location_type_keyword', axis=1)
df1['ip_location_type_keyword_1'] = t1
df1['ip_location_type_keyword_2'] = t2
df1['ip_location_type_keyword_3'] = t3
df1['ip_location_type_keyword_4'] = t4

# ip_risk_level(IP威胁级别):replace
df1['ip_risk_level'] = df1['ip_risk_level'].replace({'1级': 1, '2级': 2, '3级': 3})

# location(地点):drop
df1 = df1.drop('location', axis=1)

# client_type(客户端类型):replace
df1['client_type'] = df1['client_type'].replace({'app': 0, 'web': 1})

# browser_source(浏览器来源):replace
df1['browser_source'] = df1['browser_source'].replace({'desktop': 0, 'mobile': 1})

# device_model(设备型号):drop
df1 = df1.drop('device_model', axis=1)

# os_type(操作系统类型):replace
df1['os_type'] = df1['os_type'].replace({'windows': 0, 'macOS': 1})

# os_version(操作系统版本号):drop
df1 = df1.drop('os_version', axis=1)

# browser_type(浏览器类型):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
for i in df1['browser_type']:
    if i == 'edge':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t1.append(0)
    if i == 'chrome':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t2.append(0)
    if i == 'firefox':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t3.append(0)
    if i == 'ie':
        t4.append(1)
        t5.append(0)
        continue
    else:
        t4.append(0)
    if i == 'safari':
        t5.append(1)
    else:
        t5.append(0)
df1 = df1.drop('browser_type', axis=1)
df1['browser_type_1'] = t1
df1['browser_type_2'] = t2
df1['browser_type_3'] = t3
df1['browser_type_4'] = t4
df1['browser_type_5'] = t5

# browser_version(浏览器版本号):drop
df1 = df1.drop('browser_version', axis=1)

# bus_system_code(应用系统编码):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []
t7 = []
for i in df1['bus_system_code']:
    if i == 'attendance':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t1.append(0)
    if i == 'coremail':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t2.append(0)
    if i == 'crm':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t3.append(0)
    if i == 'oa':
        t4.append(1)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t4.append(0)
    if i == 'order-mgnt':
        t5.append(1)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t5.append(0)
    if i == 'reimbursement':
        t6.append(1)
        t7.append(0)
        continue
    else:
        t6.append(0)
        if i == 'salary':
        t7.append(1)
    else:
        t7.append(0)
df1 = df1.drop('bus_system_code', axis=1)
df1['bus_system_code_1'] = t1
df1['bus_system_code_2'] = t2
df1['bus_system_code_3'] = t3
df1['bus_system_code_4'] = t4
df1['bus_system_code_5'] = t5
df1['bus_system_code_6'] = t6
df1['bus_system_code_7'] = t7

# op_target(应用系统类目):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
for i in df1['op_target']:
    if i == 'sales':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t1.append(0)
    if i == 'finance':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t2.append(0)
    if i == 'management':
        t3.append(1)
        t4.append(0)
        continue
    else:
        t3.append(0)
    if i == 'hr':
        t4.append(1)
    else:
        t4.append(0)
df1 = df1.drop('op_target', axis=1)
df1['op_target_1'] = t1
df1['op_target_2'] = t2
df1['op_target_3'] = t3
df1['op_target_4'] = t4

# risk_label(风险标识):\

'''
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
print(df1)
'''

# 写入excel
df1.to_excel('train.xlsx')

# test_dataset(测试集)
df2 = pd.read_csv('test_dataset.csv', sep='\t')

# session_id(认证ID):drop
df2 = df2.drop('session_id', axis=1)

# op_date(认证时间):(as follows)
t = []
for idx in range(0, len(df2)):
    i = df2.loc[idx, :]
    tmp = df2[(df2.user_name == i['user_name']) & (df2.ip == i['ip']) & (pd.to_datetime(
        df2.op_date) < pd.to_datetime(i['op_date']))]
    if len(tmp) == 0:
        t.append(100)
    else:
        min = (pd.to_datetime(i['op_date']) - (pd.to_datetime(tmp['op_date'])).max()).total_seconds() / 60
        min = 100 if min > 100 else min
        t.append(min)
df2['op_timedelta'] = t
df2 = df2.drop('op_date', axis=1)

# user_name(用户名):drop
df2 = df2.drop('user_name', axis=1)

# action(操作类型):replace
df2['action'] = df2['action'].replace({'login': 0, 'sso': 1})

# auth_type(首次认证方式):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []  # auth_type == None
for i in df2['auth_type']:
    if i == 'pwd':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t1.append(0)
    if i == 'sms':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t2.append(0)
    if i == 'otp':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t3.append(0)
    if i == 'qr':
        t4.append(1)
        t5.append(0)
    else:
        t4.append(0)
        t5.append(1)
df2 = df2.drop('auth_type', axis=1)
df2['auth_type_1'] = t1
df2['auth_type_2'] = t2
df2['auth_type_3'] = t3
df2['auth_type_4'] = t4
df2['auth_type_5'] = t5

# ip(IP地址):drop
df2 = df2.drop('ip', axis=1)

# ip_location_type_keyword(IP类型):ont-hot
t1 = []
t2 = []
t3 = []
t4 = []
for i in df2['ip_location_type_keyword']:
    if i == '家庭宽带':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t1.append(0)
    if i == '代理IP':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t2.append(0)
    if i == '内网':
        t3.append(1)
        t4.append(0)
        continue
    else:
        t3.append(0)
    if i == '公共宽带':
        t4.append(1)
    else:
        t4.append(0)
df2 = df2.drop('ip_location_type_keyword', axis=1)
df2['ip_location_type_keyword_1'] = t1
df2['ip_location_type_keyword_2'] = t2
df2['ip_location_type_keyword_3'] = t3
df2['ip_location_type_keyword_4'] = t4

# ip_risk_level(IP威胁级别):replace
df2['ip_risk_level'] = df2['ip_risk_level'].replace({'1级': 1, '2级': 2, '3级': 3})

# location(地点):drop
df2 = df2.drop('location', axis=1)

# client_type(客户端类型):replace
df2['client_type'] = df2['client_type'].replace({'app': 0, 'web': 1})

# browser_source(浏览器来源):replace
df2['browser_source'] = df2['browser_source'].replace({'desktop': 0, 'mobile': 1})

# device_model(设备型号):drop
df2 = df2.drop('device_model', axis=1)

# os_type(操作系统类型):replace
df2['os_type'] = df2['os_type'].replace({'windows': 0, 'macOS': 1})

# os_version(操作系统版本号):drop
df2 = df2.drop('os_version', axis=1)

# browser_type(浏览器类型):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
for i in df2['browser_type']:
    if i == 'edge':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t1.append(0)
    if i == 'chrome':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t2.append(0)
    if i == 'firefox':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        continue
    else:
        t3.append(0)
    if i == 'ie':
        t4.append(1)
        t5.append(0)
        continue
    else:
        t4.append(0)
    if i == 'safari':
        t5.append(1)
    else:
        t5.append(0)
df2 = df2.drop('browser_type', axis=1)
df2['browser_type_1'] = t1
df2['browser_type_2'] = t2
df2['browser_type_3'] = t3
df2['browser_type_4'] = t4
df2['browser_type_5'] = t5

# browser_version(浏览器版本号):drop
df2 = df2.drop('browser_version', axis=1)

# bus_system_code(应用系统编码):one-hot
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []
t7 = []
for i in df2['bus_system_code']:
    if i == 'attendance':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t1.append(0)
    if i == 'coremail':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t2.append(0)
    if i == 'crm':
        t3.append(1)
        t4.append(0)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t3.append(0)
    if i == 'oa':
        t4.append(1)
        t5.append(0)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t4.append(0)
    if i == 'order-mgnt':
        t5.append(1)
        t6.append(0)
        t7.append(0)
        continue
    else:
        t5.append(0)
    if i == 'reimbursement':
        t6.append(1)
        t7.append(0)
        continue
    else:
        t6.append(0)
    if i == 'salary':
        t7.append(1)
    else:
        t7.append(0)
df2 = df2.drop('bus_system_code', axis=1)
df2['bus_system_code_1'] = t1
df2['bus_system_code_2'] = t2
df2['bus_system_code_3'] = t3
df2['bus_system_code_4'] = t4
df2['bus_system_code_5'] = t5
df2['bus_system_code_6'] = t6
df2['bus_system_code_7'] = t7

# op_target(应用系统类目):one=hot
t1 = []
t2 = []
t3 = []
t4 = []
for i in df2['op_target']:
    if i == 'sales':
        t1.append(1)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t1.append(0)
    if i == 'finance':
        t2.append(1)
        t3.append(0)
        t4.append(0)
        continue
    else:
        t2.append(0)
    if i == 'management':
        t3.append(1)
        t4.append(0)
        continue
    else:
        t3.append(0)
    if i == 'hr':
        t4.append(1)
    else:
        t4.append(0)
df2 = df2.drop('op_target', axis=1)
df2['op_target_1'] = t1
df2['op_target_2'] = t2
df2['op_target_3'] = t3
df2['op_target_4'] = t4

# risk_label(风险标识):(无)

'''
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
print(df2)
'''

# 写入excel
df2.to_excel('predict.xlsx')
