# -*- coding: utf-8 -*-
"""该案例展示了一个简单的连续型决策变量最大化目标的单目标优化问题的求解。问题的定义详见MyProblem.py."""
from MyProblem import MyProblem  # 导入自定义问题接口
from sklearn.linear_model import LinearRegression
import geatpy as ea  # import geatpy
import numpy as np
import pandas as pd
import scipy.io as sio

# df = pd.read_excel("./B0.xlsx")
# data = df.values
# Bm = df.values[:, 3]
df = pd.read_excel("./ConT-003 sample points field_20240620.xlsx",sheet_name='Coordination')
Bm =df.iloc[:,7].values
AA = sio.loadmat("AA.mat")["AA"]
BB = sio.loadmat("BB.mat")["BB"].reshape(-1)
As_Bx = sio.loadmat("As_Bx.mat")["As_Bx"]
x0 = sio.loadmat("x0.mat")["x0"]
B_final = sio.loadmat("Bx_final_x0.mat")
Bx_final_data = B_final['Bx_final_x0']
Nz = 15
Np = 36
Bu = Bx_final_data.max()
Bl = Bx_final_data.min()
upper_part = Bu - Bm
lower_part = -(Bl - Bm)
BB = np.vstack((upper_part, lower_part))

Dim = Nz*Np
NIND = 2000  # 种群规模
var_set = np.array([ 0, 1 , 1.5, 2])
dic={}
for i,j in enumerate(var_set):
    dic[j] = i 

x0_index  = np.array([dic[i] for i in x0.reshape(-1)])
prophet =  np.tile(x0_index, (NIND, 1))[0]
# print(prophet.shape)
# if __name__ == '__main__':
for i in range(10):
    print(i)
    # 实例化问题对象
    problem = MyProblem(NIND, var_set, AA, BB, As_Bx ,Bm, Dim, [], [],x0)
    # problem = MyProblem()
    # 构建算法
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=1000),
        MAXGEN=100,  # 最大进化代数。
        logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
        maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=0,
                      prophet=prophet,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False)
    # print(str(round(res["ObjV"][0][0],2)))
    result=var_set[res["Vars"]]
    # print(result)
    # sio.savemat("54mm_1_result/"+str(round(res["ObjV"][0][0],2))+".mat",{'result':result})
    
    sio.savemat("result/"+str(round(res["ObjV"][0][0],2))+".mat",{'result':result})
