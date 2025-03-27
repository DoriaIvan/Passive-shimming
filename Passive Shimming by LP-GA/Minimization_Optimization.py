# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat, savemat
# from scipy.optimize import linprog
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import LinearRegression

# def minimization(B0, As_Bx, ppm_t, Nz, Np, ironThickness_min, ironThickness_max):
#     """优化磁片厚度以提高磁场均匀性
    
#     Args:
#         B0: 原始磁场数据
#         As_Bx: 磁场贡献矩阵
#         ppm_t: 目标均匀度（ppm）
#         Nz, Np: 磁片排布的垂直和方位方向网格数
#         ironThickness_min, ironThickness_max: 磁片厚度的范围
        
#     Returns:
#         t0: 优化后的磁片厚度矩阵（展平）
#     """
#     print("开始优化过程...")
    
#     # 从B0中提取磁场数据
#     try:
#         Bm = B0[:, 7]  # 单位 mT
#     except IndexError:
#         print("警告：B0数据列不足，尝试使用最后一列")
#         Bm = B0[:, -1]
    
#     print(f"读取到 {len(Bm)} 个磁场测量点")
    
#     N = Nz * Np  # 总的磁片数量
#     M = len(Bm)  # 测量点数量
    
#     # 使用常数作为参考场值
#     b_factor = np.mean(Bm)  # mT
#     print(f"参考场值: {b_factor} mT")
    
#     # 构建线性规划约束条件
#     AA = np.vstack((As_Bx, -As_Bx))  # 不等式约束左侧
#     print(f"约束矩阵AA形状: {AA.shape}")
    
#     # 保存数据
#     savemat('AA.mat', {"AA": AA})
    
#     Bu = b_factor * (1 + ppm_t / 2)  # 设置期望场的上边界
#     Bl = b_factor * (1 - ppm_t / 2)  # 设置期望场的下边界
    
#     BB = np.zeros(2 * M)
#     BB[:M] = Bu - Bm
#     BB[M:2*M] = -(Bl - Bm)
    
#     savemat('BB.mat', {"BB": BB})
    
#     # 设置厚度边界
#     lb = np.zeros(N)  # 磁片厚度下限
#     ub = np.zeros(N) + ironThickness_max  # 磁片厚度上限
    
#     # 目标函数 - 最小化磁片总量
#     f = np.ones(N)
    
#     # 线性规划求解
#     print("开始线性规划优化...")
#     try:
#         res = linprog(f, A_ub=AA, b_ub=BB, bounds=[(lb[i], ub[i]) for i in range(N)], 
#                       options={'disp': True})
        
#         if res.success:
#             print("优化成功!")
#             t0 = res.x
#             print(f"目标函数值: {res.fun}")
#         else:
#             print("优化失败:", res.message)
#             t0 = np.zeros(N)
#     except Exception as e:
#         print(f"优化过程出错: {e}")
#         t0 = np.zeros(N)
    
#     return t0

# # 主程序
# if __name__ == "__main__":
#     print("开始程序...")
    
#     try:
#         # 加载数据文件
#         print("加载数据文件...")
        
#         # 尝试不同的加载方式
#         try:
#             # 首先尝试加载.npy文件
#             data = np.load('data.npy')
#             print("成功加载data.npy")
#         except:
#             # 如果失败，尝试加载.mat文件
#             data_mat = loadmat('data.mat')
#             data = data_mat['data']
#             print("成功加载data.mat")
        
#         try:
#             # 首先尝试直接从.mat文件加载As_Bx
#             As_Bx = loadmat('As_Bx.mat')['As_Bx']
#             print("成功加载As_Bx.mat")
#         except:
#             print("无法加载As_Bx.mat，请确保此文件存在")
#             exit(1)
        
#         # 打印数据维度
#         print(f"data 形状: {data.shape}")
#         print(f"As_Bx 形状: {As_Bx.shape}")
        
#         # 获取磁场数据
#         try:
#             Bm_x = data[:, 7]  # 单位 mT
#         except IndexError:
#             print("警告：data数据列不足，尝试使用最后一列")
#             Bm_x = data[:, -1]
        
#         Bm_x_ave = (np.max(Bm_x) + np.min(Bm_x)) / 2
#         print(f"磁场平均值: {Bm_x_ave} mT")
        
#         # 设置参数
#         Nz = 15
#         Np = 36
        
#         ppm_t = 200e-6  # 设定ppm目标
        
#         ironThickness_min = 0
#         ironThickness_max = 2
        
#         print(f"参数设置: Nz={Nz}, Np={Np}, ppm_t={ppm_t}, 厚度范围=[{ironThickness_min}, {ironThickness_max}]")
        
#         # 执行优化
#         print("开始执行优化...")
#         t0 = minimization(data, As_Bx, ppm_t, Nz, Np, ironThickness_min, ironThickness_max)
        
#         # 四舍五入获取整数解
#         x0 = np.round(t0).astype(int)
#         savemat('x0.mat', {"x0": x0})
#         print("保存x0.mat完成")
        
#         # 重塑为矩阵形式
#         A_t0 = t0.reshape(Nz, Np)
#         A_x0 = x0.reshape(Nz, Np)
        
#         num_positive = np.sum(x0 > 0)
#         print(f"正值磁片数量: {num_positive}")
        
#         # 绘图1: 磁片排布矩阵
#         plt.figure(figsize=(15, 6))
#         im = plt.imshow(A_x0, cmap='jet')
#         plt.colorbar(im, ticks=range(int(ironThickness_max)+1))
        
#         # 设置坐标轴
#         plt.gca().xaxis.set_ticks_position('top')
#         plt.xlabel('Azimuthal Index', fontsize=14, fontweight='bold')
#         plt.ylabel('Vertical Index', fontsize=14, fontweight='bold')
        
#         # 在矩阵中标注数值
#         for i in range(Nz):
#             for j in range(Np):
#                 plt.text(j, i, str(A_x0[i, j]), ha='center', va='center', color='w', fontsize=12, fontweight='bold')
        
#         plt.tight_layout()
#         plt.savefig('matrix_plot.png')
#         print("保存矩阵图完成")
        
#         # 绘图2: 3D圆柱面表示
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         h = 150
#         phi = np.linspace(0, 2*np.pi, Np)
#         z = np.linspace(0, h, Nz)
#         Phi, Z = np.meshgrid(phi, z)
        
#         X_cylinder = np.cos(Phi)
#         Y_cylinder = np.sin(Phi)
        
#         surf = ax.plot_surface(X_cylinder, Y_cylinder, Z, facecolors=cm.jet(A_t0/ironThickness_max), 
#                               linewidth=1, edgecolor='k')
        
#         # 添加颜色条
#         m = cm.ScalarMappable(cmap=cm.jet)
#         m.set_array(A_t0)
#         plt.colorbar(m, ax=ax, ticks=np.linspace(0, ironThickness_max, 5))
        
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.view_init(20, 45)
#         plt.savefig('cylinder_plot.png')
#         print("保存圆柱图完成")
        
#         # 计算匀场前后的磁场
#         dBx_t0 = As_Bx @ t0
#         dBx_x0 = As_Bx @ x0
        
#         Bx_final_t0 = Bm_x + dBx_t0  # 单位 mT
#         Bx_final_x0 = Bm_x + dBx_x0
        
#         print(f"平均磁场: {np.mean(Bx_final_t0)}")
#         print(f"最大磁场: {np.max(Bx_final_t0)}")
#         print(f"最小磁场: {np.min(Bx_final_t0)}")
        
#         savemat('Bx_final_x0.mat', {"Bx_final_x0": Bx_final_x0})
#         print("保存Bx_final_x0.mat完成")
        
#         # 绘图3: 匀场前后对比
#         plt.figure(figsize=(10, 6))
#         plt.plot(Bm_x, 'r', label='Initial Field')
#         plt.plot(Bx_final_t0, 'b', label='Shimming Field(t)')
#         plt.plot(Bx_final_x0, 'k', label='Shimming Field(t_0)')
        
#         plt.xlabel('Sample Points', fontsize=12)
#         plt.ylabel('Magnetic field strength / mT', fontsize=12)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig('field_comparison.png')
#         print("保存场对比图完成")
        
#         # 尝试加载球谐数据
#         try:
#             F = np.load('F.npy')
#             print("成功加载F.npy")
#         except:
#             try:
#                 F = loadmat('F.mat')['F']
#                 print("成功加载F.mat")
#             except:
#                 print("无法加载球谐数据，跳过球谐分析部分")
#                 F = None
        
#         # 如果有球谐数据，进行球谐分析
#         if F is not None:
#             print(f"F 形状: {F.shape}")
            
#             # 打印结果
#             print(f"初始场峰峰不均匀度为 {(np.max(Bm_x) - np.min(Bm_x))*1e6/np.mean(Bm_x)} ppm")
#             print(f"被动场峰峰不均匀度(t0)为 {(np.max(Bx_final_t0) - np.min(Bx_final_t0))*1e6/np.mean(Bx_final_t0)} ppm")
#             print(f"被动场峰峰不均匀度(x0)为 {(np.max(Bx_final_x0) - np.min(Bx_final_x0))*1e6/np.mean(Bx_final_x0)} ppm")
#             print(f"磁片数量: {num_positive}")
        
#         # 显示所有图形
#         plt.show()
        
#     except Exception as e:
#         print(f"执行过程中出现错误: {e}")
#         import traceback
#         traceback.print_exc()  # 打印详细错误信息
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import linprog
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def minimization(B0, As_Bx, ppm_t, Nz, Np, ironThickness_min, ironThickness_max):
    """优化磁片厚度以提高磁场均匀性
    
    Args:
        B0: 原始磁场数据
        As_Bx: 磁场贡献矩阵
        ppm_t: 目标均匀度（ppm）
        Nz, Np: 磁片排布的垂直和方位方向网格数
        ironThickness_min, ironThickness_max: 磁片厚度的范围
        
    Returns:
        t0: 优化后的磁片厚度矩阵（展平）
    """
    print("开始优化过程...")
    
    # 从B0中提取磁场数据
    try:
        Bm = B0[:, 7]  # 单位 mT
    except IndexError:
        print("警告：B0数据列不足，尝试使用最后一列")
        Bm = B0[:, -1]
    
    print(f"读取到 {len(Bm)} 个磁场测量点")
    
    N = Nz * Np  # 总的磁片数量
    M = len(Bm)  # 测量点数量
    
    # 使用常数作为参考场值
    b_factor = np.mean(Bm)  # mT
    print(f"参考场值: {b_factor} mT")
    
    # 构建线性规划约束条件
    AA = np.vstack((As_Bx, -As_Bx))  # 不等式约束左侧
    print(f"约束矩阵AA形状: {AA.shape}")
    
    # 保存数据
    savemat('AA.mat', {"AA": AA})
    
    Bu = b_factor * (1 + ppm_t / 2)  # 设置期望场的上边界
    Bl = b_factor * (1 - ppm_t / 2)  # 设置期望场的下边界
    
    BB = np.zeros(2 * M)
    BB[:M] = Bu - Bm
    BB[M:2*M] = -(Bl - Bm)
    
    savemat('BB.mat', {"BB": BB})
    
    # 设置厚度边界
    lb = np.zeros(N)  # 磁片厚度下限
    ub = np.zeros(N) + ironThickness_max  # 磁片厚度上限
    
    # 目标函数 - 最小化磁片总量
    f = np.ones(N)
    
    # 线性规划求解
    print("开始线性规划优化...")
    try:
        res = linprog(f, A_ub=AA, b_ub=BB, bounds=[(lb[i], ub[i]) for i in range(N)], 
                      options={'disp': True})
        
        if res.success:
            print("优化成功!")
            t0 = res.x
            print(f"目标函数值: {res.fun}")
        else:
            print("优化失败:", res.message)
            t0 = np.zeros(N)
    except Exception as e:
        print(f"优化过程出错: {e}")
        t0 = np.zeros(N)
    
    return t0

def calculate_inhomogeneity(F, b_factor, Bm_x_ave):
    """计算球谐分量的不均匀度
    
    Args:
        F: 球谐矩阵
        b_factor: 球谐系数
        Bm_x_ave: 平均磁场
        
    Returns:
        H_sh: 不均匀度向量
    """
    # 这里使用矩阵乘法计算每个分量的贡献
    B_sh = np.zeros((F.shape[0], len(b_factor)))
    
    # 对于每个系数，计算其对应的磁场分量
    for i in range(len(b_factor)):
        if i == 0:  # 常数项
            B_sh[:, i] = b_factor[i]
        else:  # 其他项
            if i-1 < F.shape[1]:  # 确保不超出F的列数
                B_sh[:, i] = F[:, i-1] * b_factor[i]
    
    # 计算每个分量的不均匀度
    H_sh = np.zeros(len(b_factor))
    for i in range(len(b_factor)):
        column_data = B_sh[:, i]
        max_val = np.max(column_data)
        min_val = np.min(column_data)
        H_sh[i] = (max_val - min_val) / Bm_x_ave * 1e6
    
    return H_sh

# 主程序
if __name__ == "__main__":
    print("开始程序...")
    
    try:
        # 加载数据文件
        print("加载数据文件...")
        
        # 尝试不同的加载方式
        try:
            # 首先尝试加载.npy文件
            data = np.load('data.npy')
            print("成功加载data.npy")
        except:
            # 如果失败，尝试加载.mat文件
            data_mat = loadmat('data.mat')
            data = data_mat['data']
            print("成功加载data.mat")
        
        try:
            # 首先尝试直接从.mat文件加载As_Bx
            As_Bx = loadmat('As_Bx.mat')['As_Bx']
            print("成功加载As_Bx.mat")
        except:
            print("无法加载As_Bx.mat，请确保此文件存在")
            exit(1)
        
        # 打印数据维度
        print(f"data 形状: {data.shape}")
        print(f"As_Bx 形状: {As_Bx.shape}")
        
        # 获取磁场数据
        try:
            Bm_x = data[:, 7]  # 单位 mT
        except IndexError:
            print("警告：data数据列不足，尝试使用第8列")
            try:
                Bm_x = data[:, 7]  # 尝试第8列 (索引7)
            except:
                print("使用最后一列作为磁场数据")
                Bm_x = data[:, -1]
        
        Bm_x_ave = (np.max(Bm_x) + np.min(Bm_x)) / 2
        print(f"磁场平均值: {Bm_x_ave} mT")
        
        # 设置参数
        Nz = 15
        Np = 36
        
        ppm_t = 200e-6  # 设定ppm目标
        
        ironThickness_min = 0
        ironThickness_max = 2
        
        print(f"参数设置: Nz={Nz}, Np={Np}, ppm_t={ppm_t}, 厚度范围=[{ironThickness_min}, {ironThickness_max}]")
        
        # 执行优化
        print("开始执行优化...")
        t0 = minimization(data, As_Bx, ppm_t, Nz, Np, ironThickness_min, ironThickness_max)
        
        # 四舍五入获取整数解
        x0 = np.round(t0).astype(int)
        savemat('x0.mat', {"x0": x0})
        print("保存x0.mat完成")
        
        # 重塑为矩阵形式
        A_t0 = t0.reshape(Nz, Np)
        A_x0 = x0.reshape(Nz, Np)
        
        num_positive = np.sum(x0 > 0)
        print(f"正值磁片数量: {num_positive}")
        
        # 绘图1: 磁片排布矩阵
        plt.figure(figsize=(15, 6))
        im = plt.imshow(A_x0, cmap='jet')
        plt.colorbar(im, ticks=range(int(ironThickness_max)+1))
        
        # 设置坐标轴
        plt.gca().xaxis.set_ticks_position('top')
        plt.xlabel('Azimuthal Index', fontsize=14, fontweight='bold')
        plt.ylabel('Vertical Index', fontsize=14, fontweight='bold')
        
        # 在矩阵中标注数值
        for i in range(Nz):
            for j in range(Np):
                plt.text(j, i, str(A_x0[i, j]), ha='center', va='center', color='w', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('matrix_plot.png')
        print("保存矩阵图完成")
        
        # 绘图2: 3D圆柱面表示
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        h = 150
        phi = np.linspace(0, 2*np.pi, Np)
        z = np.linspace(0, h, Nz)
        Phi, Z = np.meshgrid(phi, z)
        
        X_cylinder = np.cos(Phi)
        Y_cylinder = np.sin(Phi)
        
        surf = ax.plot_surface(X_cylinder, Y_cylinder, Z, facecolors=cm.jet(A_t0/ironThickness_max), 
                              linewidth=1, edgecolor='k')
        
        # 添加颜色条
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(A_t0)
        plt.colorbar(m, ax=ax, ticks=np.linspace(0, ironThickness_max, 5))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(20, 45)
        plt.savefig('cylinder_plot.png')
        print("保存圆柱图完成")
        
        # 计算匀场前后的磁场
        dBx_t0 = As_Bx @ t0
        dBx_x0 = As_Bx @ x0
        
        Bx_final_t0 = Bm_x + dBx_t0  # 单位 mT
        Bx_final_x0 = Bm_x + dBx_x0
        
        print(f"平均磁场: {np.mean(Bx_final_t0)}")
        print(f"最大磁场: {np.max(Bx_final_t0)}")
        print(f"最小磁场: {np.min(Bx_final_t0)}")
        
        savemat('Bx_final_x0.mat', {"Bx_final_x0": Bx_final_x0})
        print("保存Bx_final_x0.mat完成")
        
        # 绘图3: 匀场前后对比
        plt.figure(figsize=(10, 6))
        plt.plot(Bm_x, 'r', label='Initial Field')
        plt.plot(Bx_final_t0, 'b', label='Shimming Field(t)')
        plt.plot(Bx_final_x0, 'k', label='Shimming Field(t_0)')
        
        plt.xlabel('Sample Points', fontsize=12)
        plt.ylabel('Magnetic field strength / mT', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig('field_comparison.png')
        print("保存场对比图完成")
        
        # 尝试加载球谐数据
        try:
            F = np.load('F.npy')
            print("成功加载F.npy")
        except:
            try:
                F = loadmat('F.mat')['F']
                print("成功加载F.mat")
            except:
                print("无法加载球谐数据，跳过球谐分析部分")
                F = None
        
        # 如果有球谐数据，进行球谐分析
        if F is not None:
            print(f"F 形状: {F.shape}")
            
            # 回归分析
            model_pre = LinearRegression(fit_intercept=True).fit(F, Bm_x * 1e3)
            b_factor_pre1 = np.concatenate(([model_pre.intercept_], model_pre.coef_))
            
            model_pos1 = LinearRegression(fit_intercept=True).fit(F, Bx_final_t0 * 1e3)
            b_factor_pos1 = np.concatenate(([model_pos1.intercept_], model_pos1.coef_))
            
            model_pos2 = LinearRegression(fit_intercept=True).fit(F, Bx_final_x0 * 1e3)
            b_factor_pos2 = np.concatenate(([model_pos2.intercept_], model_pos2.coef_))
            
            print(f"球谐系数形状: pre={b_factor_pre1.shape}, pos1={b_factor_pos1.shape}, pos2={b_factor_pos2.shape}")
            
            # 获取球谐分量的标签
            sh_labels = ['Z', 'X', 'Y', 'Z^2', 'ZX', 'ZY', 'X^2', 'XY', 'Z^3', 'Z^2X', 'Z^2Y', 'ZX^2', 'XYZ', 'X^3', 'Y^3']
            # 确保标签数量与数据匹配
            sh_labels = sh_labels[:F.shape[1]]
            
            # 绘图4: 球谐系数对比
            plt.figure(figsize=(12, 12))
            
            plt.subplot(3, 1, 1)
            plt.bar(range(len(b_factor_pre1)-1), b_factor_pre1[1:])
            plt.xlabel('Before shimming of Bx', fontsize=12)
            plt.ylabel('Amplitude / uT', fontsize=12)
            plt.xticks(range(len(b_factor_pre1)-1), sh_labels)
            plt.ylim([-0.1, 0.8])
            
            plt.subplot(3, 1, 2)
            plt.bar(range(len(b_factor_pos1)-1), b_factor_pos1[1:])
            plt.xlabel('After shimming of Bx(t)', fontsize=12)
            plt.ylabel('Amplitude /uT', fontsize=12)
            plt.xticks(range(len(b_factor_pos1)-1), sh_labels)
            plt.ylim([-0.1, 0.8])
            
            plt.subplot(3, 1, 3)
            plt.bar(range(len(b_factor_pos2)-1), b_factor_pos2[1:])
            plt.xlabel('After shimming of Bx(t_0)', fontsize=12)
            plt.ylabel('Amplitude /uT', fontsize=12)
            plt.xticks(range(len(b_factor_pos2)-1), sh_labels)
            plt.ylim([-0.1, 0.8])
            
            plt.tight_layout()
            plt.savefig('harmonic_coefficients.png')
            print("保存球谐系数图完成")
            
            # 计算不均匀度
            H_sh_pre1 = calculate_inhomogeneity(F, b_factor_pre1, Bm_x_ave)
            H_sh_pos1 = calculate_inhomogeneity(F, b_factor_pos1, Bm_x_ave)
            H_sh_pos2 = calculate_inhomogeneity(F, b_factor_pos2, Bm_x_ave)
            
            # 绘图5: 不均匀度对比 (与MATLAB代码对应的图)
            plt.figure(figsize=(12, 12))
            
            plt.subplot(3, 1, 1)
            plt.bar(range(len(H_sh_pre1)-1), H_sh_pre1[1:] * 1e-3)
            plt.xlabel('Before shimming of Bm', fontsize=12)
            plt.ylabel('H / ppm', fontsize=12)
            plt.xticks(range(len(H_sh_pre1)-1), sh_labels)
            plt.ylim([0, 800])
            
            plt.subplot(3, 1, 2)
            plt.bar(range(len(H_sh_pos1)-1), H_sh_pos1[1:] * 1e-3)
            plt.xlabel('After shimming of Bm (LP solution t)', fontsize=12)
            plt.ylabel('H / ppm', fontsize=12)
            plt.xticks(range(len(H_sh_pos1)-1), sh_labels)
            plt.ylim([0, 800])
            
            plt.subplot(3, 1, 3)
            plt.bar(range(len(H_sh_pos2)-1), H_sh_pos2[1:] * 1e-3)
            plt.xlabel('After shimming of Bm (rounded solution t_0)', fontsize=12)
            plt.ylabel('H / ppm', fontsize=12)
            plt.xticks(range(len(H_sh_pos2)-1), sh_labels)
            plt.ylim([0, 800])
            
            plt.tight_layout()
            plt.savefig('harmonic_inhomogeneity.png')
            print("保存不均匀度对比图完成")
            
            # 打印结果
            print(f"初始场峰峰不均匀度为 {(np.max(Bm_x) - np.min(Bm_x))*1e6/np.mean(Bm_x)} ppm")
            print(f"被动场峰峰不均匀度(t0)为 {(np.max(Bx_final_t0) - np.min(Bx_final_t0))*1e6/np.mean(Bx_final_t0)} ppm")
            print(f"被动场峰峰不均匀度(x0)为 {(np.max(Bx_final_x0) - np.min(Bx_final_x0))*1e6/np.mean(Bx_final_x0)} ppm")
            print(f"磁片数量: {num_positive}")
        
        # 显示所有图形
        plt.show()
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息