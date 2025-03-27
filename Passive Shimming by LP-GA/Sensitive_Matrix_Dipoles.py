import numpy as np
import pandas as pd
from scipy.io import savemat

def dipole_magnetic_field(x_f, y_f, z_f, x_s, y_s, z_s, m_x, m_y):
    """计算磁偶极子场"""
    x = x_f - x_s
    y = y_f - y_s
    z = z_f - z_s
    
    r_squared = x**2 + y**2 + z**2
    r_power = r_squared**(5/2)
    
    a = 3 * (m_x * x + m_y * y) * x
    b = r_squared * m_x
    
    return (a - b) / r_power

# 读取Excel数据，跳过可能的非数值行
try:
    # 首先尝试自动处理 - pandas会尝试推断哪些列是数值型
    data = pd.read_excel('F:/ProjectData/Matlab/Magnetic Dipoles PS/ConT-003号小2MHz球面坐标数据20240620.xlsx', 
                         sheet_name='Coordination')
    
    # 检查是否有包含文本的行，如果有则需要特殊处理
    if data.iloc[0].astype(str).str.contains('=').any():
        # 发现文本行，重新读取，确保跳过这些行
        print("检测到包含文本的行，尝试跳过...")
        data = pd.read_excel('F:/ProjectData/Matlab/Magnetic Dipoles PS/ConT-003号小2MHz球面坐标数据20240620.xlsx', 
                             sheet_name='Coordination', header=None, skiprows=1)
except Exception as e:
    print(f"读取时出现错误: {e}")
    print("尝试跳过第一行...")
    data = pd.read_excel('F:/ProjectData/Matlab/Magnetic Dipoles PS/ConT-003号小2MHz球面坐标数据20240620.xlsx', 
                         sheet_name='Coordination', header=None, skiprows=1)

# 将数据转换为数值型
numeric_data = data.select_dtypes(include=['number'])
if numeric_data.shape[1] < 8:  # 确保至少有8列数值数据
    print("警告：数据列少于预期，请检查Excel文件格式")

# 获取需要的数据列
data_measure = numeric_data.values
print(f"读取了 {data_measure.shape[0]} 行 {data_measure.shape[1]} 列数据")

# 确保数据列足够
if data_measure.shape[1] <= 7:
    print("错误：数据列不足，无法读取第8列的Bm值")
    exit(1)

Bm = data_measure[:, 7]  # 使用第8列作为Bm值，索引从0开始
print(f"读取了 {len(Bm)} 个Bm值")

# 场点坐标转换
if data_measure.shape[1] >= 3:
    R_f = data_measure[:, 0:3]  # 场点球面坐标
else:
    print("错误：数据列不足，无法读取场点坐标")
    exit(1)

# 转换为笛卡尔坐标系
r_f = np.zeros((len(R_f), 3))
r_f[:, 0] = R_f[:, 0] * np.sin(R_f[:, 1]) * np.cos(R_f[:, 2])  # x
r_f[:, 1] = R_f[:, 0] * np.sin(R_f[:, 1]) * np.sin(R_f[:, 2])  # y
r_f[:, 2] = R_f[:, 0] * np.cos(R_f[:, 1])                       # z

# 设置源点参数
Nz = 15              # 沿圆柱轴的铁袋网格数
Np = 36              # 水平面上的铁袋网格数
N = Nz * Np          # 垫片口袋总数

# 垂直片分布，从z轴到-z轴
z_sheet = np.arange(70e-3, -80e-3, -10e-3)
# 片材的方位分布，从0度到360度
phi_sheet_deg = np.arange(5, 365, 10)
# 磁片半径
r_sheet = 55.2e-3

# 生成网格坐标
Phi_sheet, Z_sheet = np.meshgrid(phi_sheet_deg, z_sheet)

# 所有磁片充磁方向同向
r_s = np.zeros((Phi_sheet.size, 3))
r_s[:, 0] = r_sheet * np.cos(np.deg2rad(Phi_sheet.flatten()))
r_s[:, 1] = r_sheet * np.sin(np.deg2rad(Phi_sheet.flatten()))
r_s[:, 2] = Z_sheet.flatten()

# 材料参数
Thickness = 1e-3      # 片厚
delta_phi = 3e-3      # 水平圆周围的片大小
delta_z = 2e-3        # 沿圆柱轴的铁片尺寸
u0 = 4 * np.pi * 1e-7  # 真空磁导率
ur = 1.10218          # 相对磁导率
Br = 1.2              # 单位T，一象限充磁方向径向向外

# 定义磁化方向向量 (单位A/m饱和磁化参数，材料特性，由制造商提供)
M = np.zeros((Phi_sheet.size, 3))
M[:, 0] = Br / u0 / ur * np.cos(np.deg2rad(Phi_sheet.flatten()))
M[:, 1] = Br / u0 / ur * np.sin(np.deg2rad(Phi_sheet.flatten()))

# 计算单位磁场贡献矩阵
As_Bx_unit = np.zeros((len(Bm), N))

print(f"开始计算 {len(Bm)} 个场点和 {N} 个源点之间的磁场贡献...")

# 使用向量化操作优化双循环
for p in range(len(Bm)):
    if p % 100 == 0 and p > 0:
        print(f"已完成 {p}/{len(Bm)} 个场点的计算...")
    for q in range(N):
        As_Bx_unit[p, q] = dipole_magnetic_field(
            r_f[p, 0], r_f[p, 1], r_f[p, 2],
            r_s[q, 0], r_s[q, 1], r_s[q, 2],
            M[q, 0], M[q, 1]
        )

# 计算最终磁场贡献
volume = delta_phi * delta_z * Thickness
As_Bx = u0 / (4 * np.pi) * As_Bx_unit * volume * 1e3  # mT

# 保存结果
savemat('As_Bx.mat', {'As_Bx': As_Bx})

print(f"计算完成，已保存结果到 As_Bx.mat")