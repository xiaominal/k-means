import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import  math
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
path = 'D:\Pythontest\用户分析\电商消费行为\data\\'
df = pd.read_csv(open(path+'propecess.csv'))



'''''''''数据预处理'''''''''

df= df[~df['USERID'].isin([74270])]
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
user_table = df.groupby('USERID',as_index=False)['AMOUNTINFO'].agg({'M':'sum','F':'count'})
def r(data):
    d = data.iat[-1, 1]
    deadlinedate = pd.datetime(2017, 1, 1)
    #print(type(d),type(deadlinedate))
    r=deadlinedate-d
    r =  r/np.timedelta64(1,'D')
    return r
first_mon = df.sort_values(by='ORDERDATE').groupby('USERID',as_index=False).apply(r)
user_table['R'] = first_mon
data =(user_table-user_table.mean(axis = 0))/user_table.std(axis = 0)
data = data.drop('USERID',axis=1)
data.to_csv(path+'zscoreddata_userid.csv',index=False)