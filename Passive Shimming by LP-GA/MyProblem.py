# -*- coding: utf-8 -*-
import numpy as np

import geatpy as ea

class MyProblem(ea.Problem):  # 继承Problem父类
    """Demo.

    max f = x * np.sin(10 * np.pi * x) + 2.0
    s.t.
    -1 <= x <= 2
    """
    def __init__(self, NIND, var_set, AA, BB, As_Bx, Bm,Dim, bl, bu,x0):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        self.NIND = NIND
        self.var_set = var_set  # 设定一个集合，要求决策变量的值取自于该集合
        self.AA = AA
        self.BB = BB
        self.As_Bx = As_Bx
        self.Bm = Bm
        self.bl = bl
        self.bu = bu
        self.x0=x0
        self.flag = 0
        dic={}
        for i,j in enumerate(var_set):
            dic[j] = i 

        self.x0_index  = np.array([dic[i] for i in x0.reshape(-1)])

        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0]*Dim  # 决策变量下界
        ub = [3]*Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)


    def evalVars(self, x):  # 目标函数
        x[0]= self.x0_index
        data = self.var_set[x]
        dBz = data@self.As_Bx.T
        B_final = self.Bm + dBz
        # data[0]=self.x0.reshape(-1)
        # 按行求和
        # f = np.sum(data, axis=1).reshape(-1, 1)
        #取B_final每行的最大值
        max_B_final = np.max(B_final, axis=1)
        min_B_final = np.min(B_final, axis=1)
        mean_B_final = np.mean(B_final, axis=1)
        f1= ((max_B_final - min_B_final) * 1e6 / mean_B_final).reshape(-1,1)
        print(min(f1))

        f_list=self.AA@data.T-self.BB.reshape(-1,1)


        CV = f_list.T

        return f1, CV