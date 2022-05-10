#google drive 폴더 접근
from google.colab import drive
drive.mount('/content/gdrive')

#버터 환경데이터 B양액(df_b), D양액(df_d) 불러오기
import pandas as pd
df_b = pd.read_excel('/content/gdrive/MyDrive/Agtech/project_group_F/data/data_b.xlsx')
df_d = pd.read_excel('/content/gdrive/MyDrive/Agtech/project_group_F/data/data_d.xlsx')

df_b.head(5)

df_d.head(5)

print('----버터 B양액 데이터 정보----')
df_b.info()
print('  ')
print('----버터 D양액 데이터 정보----')
df_d.info()

print('----버터 B양액 데이터 요약 정보----')
df_b.describe()

print('----버터 D양액 데이터 요약 정보----')
df_d.describe()

"""#### > 일자별 데이터 평균 구하기"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# B양액 데이터 일자별 평균값

df_b['날짜'] = pd.to_datetime(df_b['날짜'])
df_b = df_b.set_index('날짜')
df_b.info()

new_df_b = df_b.resample(rule='D').mean()

new_df_b

# D양액 데이터 일자별 평균값

df_d['날짜'] = pd.to_datetime(df_d['날짜'])
df_d = df_d.set_index('날짜')
df_d.info()

new_df_d = df_d.resample(rule='D').mean()

#B양액 데이터와 로우수 맞추기
 new_df_d.drop(index=['2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28', '2022-04-29', '2022-04-30'], inplace=True)

new_df_d

del new_df_b['Unnamed: 0']
new_df_b

del new_df_d['Unnamed: 0']
new_df_d

new_df_b.reset_index(drop = False, inplace=True)

new_df_b

new_df_d.reset_index(drop = False, inplace=True)

new_df_d

# 양액별 온도 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax1 = plt.subplot(211)
ax2 = ax1.twinx()
a, = ax1.plot(new_df_b['날짜'], new_df_b['온도(°C)'], "-", color='red', label='B-Temperature(°C)')
b, = ax2.plot(new_df_d['날짜'], new_df_d['온도(°C)'], "-", color='blue', label='D-Temperature(°C)')
p = [a, b]
ax1.legend(p, [p_.get_label() for p_ in p])
plt.grid()
plt.show()

# 양액별 습도 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax3 = plt.subplot(211)
ax4 = ax3.twinx()
c, = ax3.plot(new_df_b['날짜'], new_df_b['습도(%)'], "-", color='red', label='B-Humidity(%)')
d, = ax4.plot(new_df_d['날짜'], new_df_d['습도(%)'], "-", color='blue', label='D-Humidity(%)')
q = [c, d]
ax3.legend(q, [q_.get_label() for q_ in q])
plt.grid()
plt.show()

# 양액별 CO2 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax5 = plt.subplot(211)
ax6 = ax5.twinx()
e, = ax5.plot(new_df_b['날짜'], new_df_b['CO2'], "-", color='red', label='B-CO2')
f, = ax6.plot(new_df_d['날짜'], new_df_d['CO2'], "-", color='blue', label='D-CO2')
r = [e, f]
ax5.legend(r, [r_.get_label() for r_ in r])
plt.grid()
plt.show()

# 양액별 EC 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax7 = plt.subplot(211)
ax8 = ax7.twinx()
g, = ax7.plot(new_df_b['날짜'], new_df_b['EC'], "-", color='red', label='B-EC')
h, = ax8.plot(new_df_d['날짜'], new_df_d['EC'], "-", color='blue', label='D-EC')
s = [g, h]
ax7.legend(s, [s_.get_label() for s_ in s])
plt.grid()
plt.show()

# 양액별 pH 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax9 = plt.subplot(211)
ax10 = ax9.twinx()
i, = ax9.plot(new_df_b['날짜'], new_df_b['pH'], "-", color='red', label='B-pH')
j, = ax10.plot(new_df_d['날짜'], new_df_d['pH'], "-", color='blue', label='D-pH')
t = [i, j]
ax9.legend(t, [t_.get_label() for t_ in t])
plt.grid()
plt.show()

# B양액 CO2 vs EC 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax11 = plt.subplot(211)
ax12 = ax11.twinx()
k, = ax11.plot(new_df_b['날짜'], new_df_b['EC'], "-", color='orange', label='B-EC')
l, = ax12.plot(new_df_b['날짜'], new_df_b['CO2'], "-", color='purple', label='B-CO2')
u = [k, l]
ax11.legend(u, [u_.get_label() for u_ in u])
plt.grid()
plt.show()

# D양액 CO2 vs EC 비교
plt.figure(figsize=(13,6))
plt.xticks(rotation=90)
ax13 = plt.subplot(211)
ax14 = ax13.twinx()
m, = ax13.plot(new_df_d['날짜'], new_df_d['EC'], "-", color='orange', label='D-EC')
n, = ax14.plot(new_df_d['날짜'], new_df_d['CO2'], "-", color='purple', label='D-CO2')
v = [m, n]
ax11.legend(v, [v_.get_label() for v_ in v])
plt.grid()
plt.show()

import seaborn as sns
comparison_b = new_df_b[['온도(°C)', '습도(%)', 'CO2', 'EC', 'pH']].dropna().copy()

plt.figure(figsize=(5,5))
plt.title("Section B", fontsize=15)
sns.heatmap(data = comparison_b.corr(),
            annot=True,
            fmt = '.2f', linewidths=.5, cmap='coolwarm',
            vmin = -1, vmax = 1, center = 0)
plt.show()

comparison_d = new_df_d[['온도(°C)', '습도(%)', 'CO2', 'EC', 'pH']].dropna().copy()

plt.figure(figsize=(5,5))
plt.title("Section D", fontsize=15)
sns.heatmap(data = comparison_d.corr(),
            annot=True,
            fmt = '.2f', linewidths=.5, cmap='coolwarm',
            vmin = -1, vmax = 1, center = 0)
plt.show()