import shapefile  # shapefile库安装：pip install pyshp
import numpy as np
import dgl
import torch
import math
import sys
import matplotlib.pyplot as plt
import networkx as nx
import xlwt, xlrd
from xlutils.copy import copy
from enum import Enum
from dgl.nn import GATConv

#定义卷积类型
class GCNType(Enum):
    GraphCon = 1    
    TAGCon = 2
    SAGECon = 3
    GATcon = 4

#定义超参数
intK = 2#TAGCN的K参数
intLayerCount = 7#卷积层的数目
intEpoch = 100#训练轮次
intVectorLength = 256#图嵌入向量维度
typeGCN = GCNType.TAGCon #选择具体卷积类型
intBatchSize = 64#batch数目




#用来记录实验数据
workbookname = 'TAGCN.xls'
if typeGCN == GCNType.GraphCon:
    workbookname = 'GCN.xls'
if typeGCN == GCNType.SAGECon:
    workbookname = 'SAGECon.xls'
if typeGCN == GCNType.GATcon:
    workbookname = 'GATcon.xls'
rdbk = xlrd.open_workbook(workbookname, formatting_info=True)
sheetname = str(intLayerCount) + '-' + str(intVectorLength) + '-' + str(intK) + '-nn-32-300-p'
#workbook = xlwt.Workbook()
wtbk =  copy(rdbk)
worksheet = wtbk.add_sheet(sheetname, cell_overwrite_ok=False)

#数据源
#filename = 'test_5010r'
#filename = 'test_5010x'
filename = 'test_5010'
#filename1 = 'test_5010_zengmi'
file = './data/' + filename + '.shp'
data = shapefile.Reader(file)
"""根据data的record属性可以取到每一个形状的标签，如Record #5009: [1, 'Z', 5009, 1, 0]"""
# print(data.fields)
data = data.shapes()
center_points = np.genfromtxt("centre_point.content", dtype=np.float64)
center_points = center_points.reshape(-1, 2)
# print(center_points)
# print(center_points[0])
# print(center_points[0][0])
# print(center_points[0][1])
# print(center_points.shape)
# print('data的长度：', len(data))
graph = []
for i in range(len(data)):  # data的长度： 5010
    point = data[i].points  # 取data中的每一个形状，并赋值给point进行存储
    # print(point)  # 得到的point就是每一个形状的坐标点，大小不统一
    # print(len(point))  # 每个子列表的长度
    uu = []
    vv = []
    for j in range(len(point)-1):
        uu.append(j)
        if j + 1 in range(len(point)-1):
            vv.append(j+1)
        else:
            vv.append(0)
    # print("uu:", uu)
    # print("vv:", vv)
    u = np.concatenate([uu, vv])
    v = np.concatenate([vv, uu])
    # print("u:", u)
    # print("v:", v)
    # 构建图结构
    g = dgl.graph((u, v))
    # print("图g的结构形式：", g)  # g是整个图Graph(num_nodes=12, num_edges=24,
    # fig, ax = plt.subplots()
    # nx.draw(g.to_networkx(), ax=ax)  # 展示图的样式
    # plt.show()

    l_node_feature = []  # 用于存储边长特征1
    angle_feature = []  # 用于存储方位角2
    steering_angle = []  # 用于存储转向角3
    l_point_to_point = []  # 用于存储线段的中点到建筑物中心点的长度4
    l_centre_to_x1 = []  # 用于存储建筑物中心点到线段起点x1的长度5
    l_centre_to_x2 = []  # 用于存储建筑物中心点到线段终点x2的长度6
    area_x0_x1_x2 = []  # 用于存储建筑物中心点x0，线段的起始点x1以及线段的终点x2三者围成的面积7

    azimuth_angle = []  # 存储x0_x3边的方位角
    angle_x1_x0_x2 = []  # 用于存储角x0_x3的转向角的度数8
    """建筑物的中心点坐标(x0, y0)"""
    x0 = center_points[i][0]
    y0 = center_points[i][1]

    """将坐标点的排列顺序按逆时针排列，原有的坐标顺序是顺时针（向量行走方向以逆时针为准）"""
    point = point[::-1]

    for idx in range(len(point)-1):
        # 线段的两个端点坐标
        x1 = point[idx][0]  # 第一个点
        y1 = point[idx][1]  # 第一个点
        x2 = point[idx+1][0]  # 第二个点
        y2 = point[idx+1][1]  # 第二个点

        # 线段的中点坐标
        x3 = (x1 + x2) / 2.
        y3 = (y1 + y2) / 2.
        # print("xo:", x0)
        # print("yo:", y0)
        # print("x3:", x3)
        # print("y3:", y3)

        """计算线段的中点到建筑物中心点的长度l_midpoint_line_central_point"""
        l_midpoint_to_central_point = math.sqrt((x0 - x3) ** 2 + (y0 - y3) ** 2)

        """计算建筑物中心点到线段起点x1的长度"""
        temp_l_centre_to_x1 = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        """计算建筑物中心点到线段起点x2的长度"""
        temp_l_centre_to_x2 = math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)

        # 线段的长度l_node
        l_node = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        """计算三角形x0_x1_x2的面积"""
        a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        b = math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)
        c = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        p = (a + b + c) / 2.0
        s = math.sqrt(p * (p - a) * (p - b) * (p - c))

        """定义边与边之间的转向角, 转向角的范围（-180,180）之间，逆时针为正，顺时针为负"""
        if x2 == x1 and y2 > y1:
            angle = 0.
        elif x2 == x1 and y2 < y1:
            angle = 180.
        elif y1 == y2 and x1 < x2:
            angle = 90.
        elif y1 == y2 and x1 > x2:
            angle = 270.
        else:
            angle = math.atan((abs(y2 - y1) / abs(x2 - x1)))
            angle = math.degrees(angle)
            if x2 > x1 and y2 > y1:
                angle = (90. - angle)
            elif x2 > x1 and y2 < y1:
                angle = (angle + 90.)
            elif x2 < x1 and y2 < y1:
                angle = (270. - angle)
            elif x2 < x1 and y2 > y1:
                angle = (270. + angle)

        """定义边x0_x3之间的转向角, 转向角的范围（-180,180）之间，逆时针为正，顺时针为负"""
        if x3 == x0 and y2 > y0:
            angle2 = 0.
        elif x3 == x0 and y3 < y0:
            angle2 = 180.
        elif y0 == y3 and x0 < x3:
            angle2 = 90.
        elif y0 == y3 and x0 > x3:
            angle2 = 270.
        else:
            angle2 = math.atan((abs(y3 - y0) / abs(x3 - x0)))
            angle2 = math.degrees(angle2)
            if x3 > x0 and y3 > y0:
                angle2 = (90. - angle2)
            elif x3 > x0 and y3 < y0:
                angle2 = (angle2 + 90.)
            elif x3 < x0 and y3 < y0:
                angle2 = (270. - angle2)
            elif x3 < x0 and y3 > y0:
                angle2 = (270. + angle2)

        azimuth_angle.append(angle2)
        l_node_feature.append(l_node)
        angle_feature.append(angle)
        l_point_to_point.append(l_midpoint_to_central_point)
        l_centre_to_x1.append(temp_l_centre_to_x1)
        l_centre_to_x2.append(temp_l_centre_to_x2)
        area_x0_x1_x2.append(s)

    # print("中点到中点：", l_point_to_point)

    for idx in range(len(angle_feature)):  # len(angle_feature)=
        if idx == (len(angle_feature) - 1):
            temp = -1 * (angle_feature[0] - angle_feature[-1])  # 乘以-1，保证逆时针为正
            if temp < -180.:
                temp += 360.
        else:
            temp = -1. * (angle_feature[idx + 1] - angle_feature[idx])  # 乘以-1，保证逆时针为正
            if temp < -180.:
                temp += 360.

        steering_angle.append(temp)

    for idx in range(len(azimuth_angle)):  # len(angle_feature)=
        if idx == (len(azimuth_angle) - 1):
            temp = -1 * (azimuth_angle[0] - azimuth_angle[-1])  # 乘以-1，保证逆时针为正
            if temp < -180.:
                temp += 360.
        else:
            temp = -1. * (azimuth_angle[idx + 1] - azimuth_angle[idx])  # 乘以-1，保证逆时针为正
            if temp < -180.:
                temp += 360.
        angle_x1_x0_x2.append(temp)

    # print("angle_x1_x0_x2:", angle_x1_x0_x2)

    """边长归一化"""
    l_node_feature = np.array(l_node_feature)
    l_node_feature = l_node_feature / l_node_feature.sum()
    l_node_feature = l_node_feature.reshape(-1, 1)

    """方位角归一化，重塑"""
    angle_feature = np.array(angle_feature)
    angle_feature = angle_feature / 360.
    angle_feature = angle_feature.reshape(-1, 1)
    # # 将边长和角度两个数组拼接起来
    # total_feature = np.concatenate((l_node_feature, angle_feature), axis=1)

    """转向角归一化"""
    steering_angle = np.array(steering_angle, dtype=np.float64)
    steering_angle = steering_angle / steering_angle.sum()
    # print("steering_angle:", steering_angle)
    steering_angle = steering_angle.reshape(-1, 1)

    """线段的中点到建筑物中心点的长度以及归一化"""
    l_point_to_point = np.array(l_point_to_point, dtype=np.float64)
    l_point_to_point = l_point_to_point / l_point_to_point.sum()
    l_point_to_point = l_point_to_point.reshape(-1, 1)
    # print("l_point_to_point:",l_point_to_point)

    """建筑物中心点到线段起点x1的长度以及归一化"""
    l_centre_to_x1 = np.array(l_centre_to_x1, dtype=np.float64)
    l_centre_to_x1 = l_centre_to_x1 / l_centre_to_x1.sum()
    l_centre_to_x1 = l_centre_to_x1.reshape(-1, 1)

    """建筑物中心点到线段起点x2的长度以及归一化"""
    l_centre_to_x2 = np.array(l_centre_to_x2, dtype=np.float64)
    l_centre_to_x2 = l_centre_to_x2 / l_centre_to_x2.sum()
    l_centre_to_x2 = l_centre_to_x2.reshape(-1, 1)

    """中心点到线段中点的转向角以及归一化"""
    angle_x1_x0_x2 = np.array(angle_x1_x0_x2, dtype=np.float64)
    angle_x1_x0_x2 = angle_x1_x0_x2 / 360.
    angle_x1_x0_x2 = angle_x1_x0_x2.reshape(-1, 1)

    """三角形x0_x1_x2面积以及归一化"""
    area_x0_x1_x2 = np.array(area_x0_x1_x2, dtype=np.float64)
    area_x0_x1_x2 = area_x0_x1_x2 / area_x0_x1_x2.sum()
    area_x0_x1_x2 = area_x0_x1_x2.reshape(-1, 1)

    """将边长和转向角拼接起来"""
    total_feature = np.concatenate((l_node_feature, steering_angle, l_point_to_point,
                                    angle_x1_x0_x2, l_centre_to_x1, l_centre_to_x2, area_x0_x1_x2), axis=1)  # , l_centre_to_x1, l_centre_to_x2, area_x0_x1_x2
    # 添加节点属性：维度：节点数量*2
    g.ndata['x'] = torch.tensor(total_feature, dtype=torch.float64).view(-1, 7)
    graph.append(g)
    # # print("g的节点属性：", g.ndata)
    # print("g的节点属性：", g.ndata['x'])
    # # print("g的节点数量：", g.num_nodes())
    # print("g的节点属性的形状：", (g.ndata['x']).shape)

"""
标签：形式是numpy数组，顺序和test5010的顺序对应
"""
label = np.genfromtxt("label.content", dtype=np.float64)
# print(label)              # [3. 3. 3. ... 2. 9. 1.]
# print(type(label))        # <class 'numpy.ndarray'>

"""
5010个数据的建立
图存储在列表totalset中，存储形式：[(graph, label, graph_id),(graph, label, id),(graph, label, id),...(graph, label, id)]
"""
totalset = []
for i in range(len(graph)):  # for i in range(len(graph)):
    temp = (graph[i], label[i], float(i))
    totalset.append(temp)
# print("训练集：", totalset)
print("训练集的长度：", len(totalset))

"""将5000个数据按类别分开，每组500个，共10类，加上最后10个标准型"""
label_l_0, label_z_1, label_o_2, label_y_3, label_i_4, label_t_5, label_e_6, label_u_7, label_f_8, label_h_9 = [], \
                            [], [], [], [], [], [], [], [], []
# 定义一个列表，存储10类列表的名字
name_class_10 = [label_l_0, label_z_1, label_o_2, label_y_3, label_i_4, label_t_5, label_e_6, label_u_7, label_f_8, label_h_9]


for i in range(len(totalset)-10):
    if int(totalset[i][1]) == 0:
        label_l_0.append(totalset[i])
    if int(totalset[i][1]) == 1:
        label_z_1.append(totalset[i])
    if int(totalset[i][1]) == 2:
        label_o_2.append(totalset[i])
    if int(totalset[i][1]) == 3:
        label_y_3.append(totalset[i])
    if int(totalset[i][1]) == 4:
        label_i_4.append(totalset[i])
    if int(totalset[i][1]) == 5:
        label_t_5.append(totalset[i])
    if int(totalset[i][1]) == 6:
        label_e_6.append(totalset[i])
    if int(totalset[i][1]) == 7:
        label_u_7.append(totalset[i])
    if int(totalset[i][1]) == 8:
        label_f_8.append(totalset[i])
    if int(totalset[i][1]) == 9:
        label_h_9.append(totalset[i])

# print("===========》", len(label_l_0))
# print("===========》", len(label_z_1))
# print("===========》", len(label_h_9))

for list_name in name_class_10:
    # 将每个类别中的元素乱序
    np.random.shuffle(list_name)

"""重新生成新的数据集，取前300个作为训练集，100作为验证集，100作为测试集"""
new_trainset, new_testset, new_valideset = [], [], []
for list_name in name_class_10:
    new_trainset.extend(list_name[:500])
    new_testset.extend(list_name[400:500])
    new_valideset.extend(list_name[:500])

"""10个标准型"""
standard_type = totalset[5000:]  # 5000-5009

new_valideset.extend(standard_type)


"""测试集的建立"""

# 创造训练集和测试集
# np.random.shuffle(totalset)

# trainset = totalset[:3000]
# testset1 = totalset[3000:4000]
# testset2 = totalset[3500:4000]
# testset3 = totalset[4000:4500]
# testset4 = totalset[4500:5000]
# testset = [testset1]  # , testset2, testset3, testset4
# print(len(trainset))
# print(len(testset))
"""图的可视化"""
# for i in range(1, 100,3):
#     graph_show, l = trainset[i]
#     fig, ax = plt.subplots()
#     nx.draw(graph_show.to_networkx(), ax=ax)   # 将图转为networkx形式
#     ax.set_title('Class: {}'.format(l))
#     plt.show()

from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import TAGConv
from dgl.nn import SAGEConv

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout):
        super(Classifier, self).__init__()
        if typeGCN == GCNType.GraphCon:
            self.conv1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
            self.conv2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
            # self.dropout = dropout                          # 定义dropout
            self.conv3 = GraphConv(hidden_dim, hidden_dim)  # 定义第3层图卷积
            self.conv4 = GraphConv(hidden_dim, hidden_dim)  # 定义第4层图卷积
            self.conv5 = GraphConv(hidden_dim, hidden_dim)  # 定义第5层图卷积
            self.conv6 = GraphConv(hidden_dim, hidden_dim)  # 定义第6层图卷积
            self.conv7 = GraphConv(hidden_dim, hidden_dim)  # 定义第7层图卷积
            self.conv8 = GraphConv(hidden_dim, hidden_dim)  # 定义第8层图卷积
            self.conv9 = GraphConv(hidden_dim, hidden_dim)  # 定义第9层图卷积
            self.conv10 = GraphConv(hidden_dim, hidden_dim)  # 定义第10层图卷积
            self.conv11 = GraphConv(hidden_dim, hidden_dim)  # 定义第11层图卷积
            self.conv12 = GraphConv(hidden_dim, hidden_dim)  # 定义第12层图卷积
            self.conv13 = GraphConv(hidden_dim, hidden_dim)  # 定义第13层图卷积
            self.conv14 = GraphConv(hidden_dim, hidden_dim)  # 定义第14层图卷积
            self.conv15 = GraphConv(hidden_dim, hidden_dim)  # 定义第15层图卷积

        if typeGCN == GCNType.TAGCon:
            self.conv1 = TAGConv(in_dim, hidden_dim, k=intK)  # 定义第一层图卷积
            self.conv2 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第二层图卷积
            self.conv3 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第3层图卷积
            self.conv4 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第4层图卷积
            self.conv5 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第5层图卷积
            self.conv6 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第6层图卷积
            self.conv7 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第7层图卷积
            self.conv8 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第8层图卷积
            self.conv9 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第9层图卷积
            self.conv10 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第10层图卷积
            self.conv11 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第11层图卷积
            self.conv12 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第12层图卷积
            self.conv13 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第13层图卷积
            self.conv14 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第14层图卷积
            self.conv15 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第15层图卷积
            self.conv16 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第16层图卷积
            self.conv17 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第17层图卷积
            self.conv18 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第18层图卷积
            self.conv19 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第19层图卷积
            self.conv20 = TAGConv(hidden_dim, hidden_dim, k=intK)  # 定义第20层图卷积

        if typeGCN == GCNType.SAGECon:
            self.conv1 = SAGEConv(in_dim, hidden_dim, 'pool')
            self.conv2 = SAGEConv(hidden_dim, hidden_dim, 'pool')
            self.conv3 = SAGEConv(hidden_dim, hidden_dim, 'pool')
            self.conv4 = SAGEConv(hidden_dim, hidden_dim, 'pool')
            self.conv5 = SAGEConv(hidden_dim, hidden_dim, 'pool')

        if typeGCN == GCNType.GATcon:
            self.conv1 = GATConv(in_dim, hidden_dim, num_heads=3)
            self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=3)
            self.conv3 = GATConv(hidden_dim, hidden_dim, num_heads=3)
            self.conv4 = GATConv(hidden_dim, hidden_dim, num_heads=3)

        self.classify = nn.Linear(hidden_dim, n_classes)   # 定义分类器

    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        # h = g.in_degrees().view(-1, 1).float()  # [N, 1] N行1列
        # print('第一个h：', h.shape)
        h = g.ndata['x'].float()  # [N, 2]
        # print("forward中的g一次：", g)
        # print("h的形状：", h.shape)  # h的形状： torch.Size([5902, 1])

        # 执行图卷积和激活函数

        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv3(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv4(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv5(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv6(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv7(g, h))  # [N, hidden_dim]
        """h = F.relu(self.conv8(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv9(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv10(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv11(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv12(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv13(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv14(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv15(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv16(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv17(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv18(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv19(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv20(g, h))  # [N, hidden_dim]"""


        # print('第3个h：', h.shape)
        g.ndata['x'] = h  # 将特征赋予到图的节点
        # print("forward中的g二次：", g)
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'x')   # [n, hidden_dim]  n=batch_size=500  (500,256)

        return hg, self.classify(hg)  # [n, n_classes]  返回分类前的值，在测试的时候接收，进行可视化

import torch.optim as optim
from torch.utils.data import DataLoader


def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]

    graphs, labels, graph_id = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long), graph_id

# 用pytorch的DataLoader和之前定义的collect函数
data_loader = DataLoader(new_trainset, batch_size=intBatchSize, shuffle=True,
                         collate_fn=collate)
test_loader = DataLoader(new_testset, batch_size=intBatchSize, shuffle=False,
                         collate_fn=collate)

# 构造模型
model = Classifier(7, intVectorLength, 10, 0.25)
# 定义分类交叉熵损失
loss_func = nn.CrossEntropyLoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, )  # weight_decay=1e-4

def t_sne(X, label):

    import matplotlib.pyplot as plt
    from sklearn import manifold

    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.Embeddeddatadimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(10, 10))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

def distance_between_two_features(feature_1, feature_2):#定义了两个特征之间距离的方法
    distan = 0.
    assert len(feature_1) == len(feature_2)#两距离长度相等时才能进行特征距离的计算
    for i in range(0, len(feature_1)):
        distan += pow((feature_1[i]-feature_2[i]), 2)
    return math.sqrt(distan)

# 模型训练
from sklearn.metrics import accuracy_score
epoch_losses = []
epoch_accu = []
for epoch in range(intEpoch):
    model.train()
    epoch_loss = 0
    for iter, (batchg, label, graph_id) in enumerate(data_loader):
        # hg是隐藏层，即分类前最后一层
        hg, prediction = model(batchg)

        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    
    worksheet.write(epoch,0,str(epoch))
    worksheet.write(epoch,1,str(round(epoch_loss, 4)))

    epoch_losses.append(epoch_loss)

    #if (epoch % 10 == 9):
    model.eval()
    test_pred_temp, test_label_temp = [], []
    with torch.no_grad():
        for it, (batchg, label, graph_id) in enumerate(test_loader):
            test_hg, prediction = model(batchg)
            pred_t = torch.softmax(prediction, 1)
            pred_t = torch.max(pred_t, 1)[1].view(-1)
            test_pred_temp += pred_t.detach().cpu().numpy().tolist()
            test_label_temp += label.cpu().numpy().tolist()
    acc = accuracy_score(test_label_temp, test_pred_temp)
    worksheet.write(epoch,2,str(round(acc, 4)))
    print('Epoch {}, loss {:.4f}   accuracy: {}'.format(epoch, epoch_loss, acc))
    epoch_accu.append(acc)

#draw loss and acc curve
"""
plt.plot(epoch_losses)
plt.show()
plt.plot(epoch_accu)
plt.show()
"""


"""模型的验证"""
from sklearn.metrics import accuracy_score

test_pred, test_label, test_hidden, graph_id__ = [], [], [], []

validate_loader = DataLoader(new_valideset, batch_size=64, shuffle=False,
                         collate_fn=collate)
model.eval()

with torch.no_grad():
    for it, (batchg, label, graph_id) in enumerate(validate_loader):
        # print("it:", it)
        # print("测试集label:", label)
        test_hg, prediction = model(batchg)
        """隐藏层test_hidden"""
        test_hidden += test_hg.detach().cpu().numpy().tolist()
        # print("隐藏层：", test_hidden)
        pred = torch.softmax(prediction, 1)
        pred = torch.max(pred, 1)[1].view(-1)
        # print("预测值：", pred)
        test_pred += pred.detach().cpu().numpy().tolist()
        # print("预测值的第二形状：", test_pred)
        test_label += label.cpu().numpy().tolist()
        # print("test_label:", test_label)
        graph_id__.extend(graph_id)

# for i in range(len(test_hidden)):
#     temp = torch.cat([test_hidden[i], test_hidden[i+1]], 0)


# print("test_hidden的长度：", len(test_hidden))  # test_hidden的长度： 1000
# print("test_hidden的类别：", type(test_hidden))  # test_hidden的类别： <class 'list'>
# print("test_hidden的类别", type(test_hidden))
# print("test_hidden的第一个元素：", test_hidden[0])
# print("test_hidden的第一个元素的形状：", test_hidden[0].shape)
# print("test_hidden：", test_hidden)
print("accuracy: ", accuracy_score(test_label, test_pred))
"""隐藏层test_hidden"""
test_hidden = np.array(test_hidden)
# print("test_hidden：", test_hidden)
# print("test_hidden的形状：", test_hidden.shape)  # test_hidden的形状： (1000, 100)
t_sne(test_hidden, test_label)

sys.exit()


#print("test_label:", test_label)
#print("test_label的长度:", len(test_label))  # test_label的长度: 1000
#print("test_pred:", test_pred)

new_valideset_label = []
for i in new_valideset:
    new_valideset_label.append(i[1])

#print("new_valideset的标签值：", new_valideset_label)

# print("graph_id:", graph_id__)
# print("graph_id的长度:", len(graph_id__))  # graph_id的长度: 1000
# print("graph_id:", type(graph_id__))      # graph_id: <class 'list'>





# print("graph_id:", graph_id__[0])
# print("graph_id:", graph_id__[1])
# print("graph_id:", graph_id__[2])
# print("graph_id:", graph_id__[3])
# print("graph_id:", graph_id__[4])

"""
有三个列表：
1. test_pred 测试集的预测值
2. test_label 测试集的标签
3. graph_id 每一个图的id（唯一值，相当于身份证）

用列表的序号进行索引，挨个比对预测值与标签是否一致，将不一致的，按照id索引，可以取出该图，以id在totalset中查找该图，进行输出，
同时输出该图的预测值和标签
"""
test_label_copy = test_label[:]
test_pred_copy = test_pred[:]

match_error = []
for j in range(len(test_label_copy)):
    temp_i = test_label_copy.pop(0)
    temp_j = test_pred_copy.pop(0)
    if temp_i == temp_j:
        pass
    else:
        match_error.append((graph_id__[j], test_pred[j], test_label[j]))

# print(match_error)


"""将匹配错误的图可视化"""
# total_data = shapefile.Reader(r'D:\gcn网络训练_v4.0\data\test_5010.shp')
# border = total_data.shapes()
# for i in range(len(match_error)):
#     border_points = border[int(match_error[i][0])].points
#     x, y = zip(*border_points)
#     fig, ax = plt.subplots()  # 生成一张图和一张子图
#     ax.set_title("graph {} predict {} label {}".format(match_error[i][0], match_error[i][1], match_error[i][2]))
#     # plt.plot(x,y,'k-') # x横坐标 y纵坐标 ‘k-’线性为黑色
#     plt.plot(x, y, color='#6666ff', label='fungis')  # x横坐标 y纵坐标 ‘k-’线性为黑色
#     ax.grid()  # 添加网格线
#     ax.axis('equal')
#     plt.show()

"""评估指标 nn, ft,st,dcg"""
nn_average, ft_average, st_average, dcg_average = 0.0, 0.0, 0.0, 0.0
nn_correct = 0
#k_tiers = (500, 500, 500, 500, 500, 500, 500, 500, 500, 500)
k_tiers = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100)  

bu_y = test_label
bu_N = len(new_valideset)  # 1010个
bu_T = len(standard_type)
#bu_tm = [5000,5001,5002,5003,5004,5005,5006,5007,5008,5009]
bu_tm = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009]

# nn
# for i in range(bu_N - 10):  # 2000次
#     shape_similarity = []
#     for j in range(bu_N - 10):  # 2000次
#         distance = distance_between_two_features(test_hidden[i], test_hidden[j])
#         shape_similarity.append((round(distance, 4), j))
#     shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])
#     nn_correct = 0
#     if bu_y[shape_similarity[0][1]] == bu_y[i]:
#         nn_correct = nn_correct + 1
# nn_average = nn_correct*1.0/(bu_N - 10) #0*1.0/5010

print('Hard work coming...!')
nn_correct = 0
for m in range(len(new_valideset) - 10):  # len(new_valideset)==1010
    shape_similarity = []
    for n in range(len(new_valideset) - 10):
        if m == n: continue
        distance = distance_between_two_features(test_hidden[m], test_hidden[n])
        shape_similarity.append((round(distance, 4), n))#保留小数点后4位，#shape_similarity的一个数据是（距离，该图形自己的标签），
    shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])#按照距离由小到大排序，对shape_similarity按照相关性由大到小排列
    if bu_y[shape_similarity[0][1]] == bu_y[m]:
            nn_correct = nn_correct + 1   
    if m % 50 == 0:
        print(m)
nn_average = nn_correct*1.0/(len(new_valideset) - 10)
print(nn_average)

print('Hard work over!')
    #print("This is shape_similarity")
    # print(shape_similarity)#shape_similarity的一个数据是（距离，离得最近的那个图形第I个图形），按照距离由小到大排序
    # print(len(shape_similarity))


for i in range(len(standard_type)):#standard_type=10  对10个标准型的结果分别进行考量
    #if i != 1:  continue
    i_tmplate, k_tier = bu_tm[i], k_tiers[i]#bu_tm:[1000,1001,1002,1003,1004,1005,1006,1007,1008,1009],k_tiers:[10个100]
    shape_similarity = []
    for j in range(len(new_valideset) - 10):  # len(new_valideset)==1010
        
        distance = distance_between_two_features(test_hidden[int(i_tmplate)], test_hidden[j])
        shape_similarity.append((round(distance, 4), j))#保留小数点后4位，#shape_similarity的一个数据是（距离，该图形自己的标签），
    shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])#按照距离由小到大排序，对shape_similarity按照相关性由大到小排列
    #print("This is shape_similarity")
    # print(shape_similarity)#shape_similarity的一个数据是（距离，离得最近的那个图形第I个图形），按照距离由小到大排序
    # print(len(shape_similarity))

    #NN
    """
    nn_correct = 0
    if bu_y[shape_similarity[0][1]] == bu_y[int(i_tmplate)]:
        nn_correct = nn_correct + 1
    nn_average = nn_average + nn_correct #round(nn_correct*1.0/bu_N, 3)
    """

    #FT
    ft_correct = 0
    for j in range(k_tier):
        if bu_y[shape_similarity[j][1]] == bu_y[int(i_tmplate)]:#bu_y是图形形状标签数组，对shape_similarity的前500个距离最近的形状标签与第i个标准型进行匹配。
            ft_correct = ft_correct + 1
    ft_average = ft_average + round(ft_correct*1.0/k_tier, 3)#对循环500次的总和数据取平均值

    #ST
    st_correct = 0
    for j in range(2*k_tier):#对前1000个数据进行匹配
        if bu_y[shape_similarity[j][1]] == bu_y[int(i_tmplate)]:
            st_correct = st_correct + 1
    st_average = st_average + round(st_correct*1.0/k_tier, 3)

    #DCG
    ann_ref, best_ref = [], []
    for j in range(bu_N-bu_T):#5010-10
        if bu_y[shape_similarity[j][1]] == bu_y[int(i_tmplate)]:
            ann_ref.append(1)
        else:
            ann_ref.append(0)
    for j in range(bu_N-bu_T):
        if j < k_tier:
            best_ref.append(1)
        else:
            best_ref.append(0)
    # print('sum1={}, sum2={}'.format(sum(ann_ref), sum(best_ref)))#500 500 对5000个实验数据都进行遍历，每类500个，共10类
    #print(ann_ref)
    #print(best_ref)
    DCG, IDCG = 0.0, 0.0
    for j in range(bu_N-bu_T):
        DCG  = DCG  + (2**ann_ref[j]-1) / math.log(j+2, 2)#Note:i from 0  对这5000个数据中对应的500的分布用列表进行表示
        IDCG = IDCG + (2**best_ref[j]-1) / math.log(j+2, 2)#Note:i from 0  前500个数据best_ref是1，后4500个是0，通过对数，相关性越高，其影响因子越大.math.log(对数，底数)
    dcg_average = dcg_average + round(DCG*1.0/IDCG, 3)#DCG
    #print(round(DCG*1.0/IDCG, 3))
worksheet.write(0,10,str(round(nn_average, 3)))
worksheet.write(0,11,str(round(ft_average / bu_T, 3)))
worksheet.write(0,12,str(round(st_average / bu_T, 3)))
worksheet.write(0,13,str(round(dcg_average / bu_T, 3)))

print("NN  = {}".format(round(nn_average, 3)))
print("FT  = {}".format(round(ft_average / bu_T, 3)))
print("ST  = {}".format(round(st_average / bu_T, 3)))
print("DCG = {}".format(round(dcg_average / bu_T, 3)))

# 评价指标准确率、召回率
test_label_copy2 = test_label[:]
test_pred_copy2 = test_pred[:]
graph_id__copy2 = graph_id__[:]
temp_list = []  # 数据格式=[(标签，预测值，id号)，(标签，预测值，id号)，...，(标签，预测值，id号)，]    总长度=1010


for i in range(len(test_label_copy2)):
    t1 = test_label_copy2.pop()
    t2 = test_pred_copy2.pop()
    t3 = graph_id__copy2.pop()
    temp_list.append((t1, t2, t3))

#print(temp_list)

true = 0
for i in range(len(temp_list)):
    if temp_list[i][0] == temp_list[i][1]:
        true += 1

acc = round(true /1010., 4)
print("acc: ", acc)
worksheet.write(0,3,str(acc))

# TP
PR_average, RR_average, F_average = 0, 0, 0
for i in range(10):
    tp, fp, fn = 0, 0, 0
    for j in range(len(temp_list)):
        if temp_list[j][0] == temp_list[j][1] == i:
            tp += 1

        if temp_list[j][0] != i and temp_list[j][1] == i:
            fp += 1

        if temp_list[j][0] == i and temp_list[j][1] != i:
            fn += 1

    PR_average += round(tp / (tp + fp), 4)
    RR_average += round(tp / (tp + fn), 4)


PR = PR_average/10.
RR = RR_average/10.
F = round((2 * PR * RR) / (PR + RR), 4)

print("PR:", PR)
print("RR:", RR)
print("F:", F)

worksheet.write(0,5,str(PR))
worksheet.write(0,6,str(RR))
worksheet.write(0,7,str(F))

# 评价指标


# 模型分类正确数
# 0-L, 1-Z, 2-O, 3-Y, 4-I, 5-T,  6-E,  7-U, 8-F,  9-H
for i in range(10):
    correct_number_of_model_classification = 0
    total_number_of_model_classifications = 0
    number_of_manual_classifications = 0
    for j in range(len(temp_list)):
        if temp_list[j][0] == temp_list[j][1] == i:
            correct_number_of_model_classification += 1

        if temp_list[j][1] == i:
            total_number_of_model_classifications += 1

        if temp_list[j][0] == i:
            number_of_manual_classifications += 1

    P = round(correct_number_of_model_classification / total_number_of_model_classifications, 4)
    R = round(correct_number_of_model_classification / number_of_manual_classifications, 4)
    F = round(2 * P * R / (P + R), 4)
    worksheet.write(i + 2, 5, str(P))
    worksheet.write(i + 2, 6, str(R))
    worksheet.write(i + 2, 7, str(F))

    print("人工分类数：{},    模型分类数：{}，   模型正确分类数：{}".format(number_of_manual_classifications,
                      total_number_of_model_classifications, correct_number_of_model_classification))
    print("类别：{}， P值：{}， R值：{}， F值：{}".format(i, P, R, F))


wtbk.save(workbookname)




