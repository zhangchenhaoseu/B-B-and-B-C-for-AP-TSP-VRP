# 靡不有初，鲜克有终
# zhangchenhaoseu@foxmail.com
# Southeast University
# 2024/2/3 14:16

# branch and cut(k-path) for Capacity Vehicle Routing Problem with Time Window
import math
from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import random

'''定义数据类。建立用于存放和调用输入data的数据结构框架'''
class Data:
    def __init__(self):  # 建立类属性
        self.customerNum = 0  # 客户点数量
        self.nodeNum = 0  # 节点总数量（客户点、起点车场、终点车场）
        self.vehicleNum = 0  # 车辆数量
        self.capacity = 0  # 车辆容量
        self.nodeId = []  # 节点的id，不含虚拟结点，从0开始
        self.customerId = []  # 客户点的id
        self.vehicleId = None  # 车辆的id
        self.corX = []  # 节点横坐标
        self.corY = []  # 节点纵坐标
        self.demand = []  # 节点需求
        self.readyTime = []  # 节点时间窗的最早时间
        self.dueTime = []  # 节点时间窗的最晚时间
        self.serviceTime = []  # 节点服务时长
        self.distanceMatrix = None  # 节点与结点之间的距离


'''读取数据。将对应的输入数据存放至对应的数据结构框架中。函数参数包括数据路径、客户点数量（客户点在1-100之间）、车辆数量、车辆容量'''
def readData(path, customerNum, vehicleNum, capacity):
    data = Data()
    data.customerNum = customerNum
    data.vehicleNum = vehicleNum
    data.vehicleId = [i for i in range(0,vehicleNum)]
    data.capacity = capacity
    data_df = pd.read_csv(path)
    # 将1个起始车场（文本数据中的第一个）+customerNum个客户点的信息存放在对应数据结构中
    for i in range(0, data.customerNum+1):
        data.nodeId.append(data_df.loc[i, 'CUST NO']-1)  # 从0开始的所有实节点,不含虚拟结点
        data.corX.append(data_df.loc[i, 'XCOORD'])
        data.corY.append(data_df.loc[i, 'YCOORD'])
        data.demand.append(data_df.loc[i, 'DEMAND'])
        data.readyTime.append(data_df.loc[i, 'READY TIME'])
        data.dueTime.append(data_df.loc[i, 'DUE TIME'])
        data.serviceTime.append(data_df.loc[i, 'SERVICE TIME'])
    # 再增加一个虚拟终点车场，并添加对应的信息
    data.corX.append(data_df.loc[0, 'XCOORD'])
    data.corY.append(data_df.loc[0, 'YCOORD'])
    data.demand.append(data_df.loc[0, 'DEMAND'])
    data.readyTime.append(data_df.loc[0, 'READY TIME'])
    data.dueTime.append(data_df.loc[0, 'DUE TIME'])
    data.serviceTime.append(data_df.loc[0, 'SERVICE TIME'])
    data.customerId = data.nodeId.copy()
    data.customerId.remove(0)
    # 节点总数为：1个起点车场+customerNum个客户点+1个终点车场
    data.nodeNum = customerNum + 2
    # 填补距离矩阵
    data. distanceMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            if i != j:
                data.distanceMatrix[i][j] = ((data.corX[i]-data.corX[j])**2+(data.corY[i]-data.corY[j])**2)**0.5
            else:
                pass
    # print("distanceMatrix:")
    # print(data.distanceMatrix)
    return data


''' --------------------Step 1，使用guroubi建立（IP）的线性松弛问题（IPr）-------------------- '''
def modeling(data,M):
    # print('______data.distanceMatrix—\n',data.distanceMatrix)
    # Step 1.1 建立模型
    IPr = Model('IPr')
    # Step 1.2 建立决策变量
    X = [[[[] for _ in range(0, data.vehicleNum)] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)]
    S = [[[] for _ in range(0, data.vehicleNum)] for _ in range(0, data.nodeNum)]
    # print("______data.nodeNum______",data.nodeNum)
    # print("______data.vehicleNum___", data.vehicleNum)
    for i in range(0,data.nodeNum):
        for j in range(0,data.nodeNum):
            if i != j:
                for k in range(0,data.vehicleNum):
                    X[i][j][k] = IPr.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"X_{i}_{j}_{k}")  # 【记得在使用BB时候松弛】
    for i in range(0, data.nodeNum):
        for k in range(0, data.vehicleNum):
            S[i][k] = IPr.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{i}_{k}")
    IPr.update()
    # Step 1.3 建立目标函数
    obj = LinExpr(0)
    for i in range(0,data.nodeNum):
        for j in range(0,data.nodeNum):
            if i != j:
                for k in range(0,data.vehicleNum):
                    obj.addTerms(data.distanceMatrix[i][j], X[i][j][k])
    IPr.setObjective(obj, sense=GRB.MINIMIZE)
    # Step 1.4 建立约束条件
    # （1）客户点服务一次约束
    cnt = 0
    for i in range(1, data.nodeNum-1):
        expr = LinExpr(0)
        for j in range(0, data.nodeNum):
            if i != j:
                for k in range(0, data.vehicleNum):
                    expr.addTerms(1, X[i][j][k])
        cnt += 1
        IPr.addConstr(expr == 1, f'C1_{cnt}')
    # （2）起点车场流出约束
    cnt = 0
    for k in range(0, data.vehicleNum):
        expr = LinExpr(0)
        for j in range(1,data.nodeNum):
            expr.addTerms(1,X[0][j][k])
        cnt += 1
        IPr.addConstr(expr == 1, f'C2_{cnt}')
    # （3）终点车场流入约束
    cnt = 0
    for k in range(0, data.vehicleNum):
        expr = LinExpr(0)
        for i in range(0,data.nodeNum-1):
            expr.addTerms(1,X[i][data.nodeNum-1][k])
        cnt += 1
        IPr.addConstr(expr == 1, f'C3_{cnt}')
    # （4）中间节点流平衡约束
    cnt = 0
    for k in range(0,data.vehicleNum):
        for h in range(1, data.nodeNum-1):
            expr = LinExpr(0)
            for i in range(0,data.nodeNum):
                if i != h:
                    expr.addTerms(1,X[i][h][k])
            for j in range(0,data.nodeNum):
                if j != h:
                    expr.addTerms(-1, X[h][j][k])
            cnt += 1
            IPr.addConstr(expr == 0, f'C4_{cnt}')
    # （5）时间戳约束1，这里注意将t_ij用c_ij进行了代替
    cnt = 0
    for k in range(0,data.vehicleNum):
        for i in range(0,data.nodeNum):
            for j in range(0,data.nodeNum):
                if i != j:
                    expr = LinExpr(0)
                    expr.addTerms(1, S[i][k])
                    expr.addTerms(M, X[i][j][k])
                    expr.addTerms(-1, S[j][k])
                    cnt += 1
                    IPr.addConstr(expr + data.distanceMatrix[i][j] <= M, f'C5_{cnt}')
    # （6）时间戳约束2
    cnt = 0
    for i in range(1,data.nodeNum-1):
        for k in range(0,data.vehicleNum):
            expr = LinExpr(0)
            expr.addTerms(1,S[i][k])
            cnt += 1
            IPr.addConstr(expr <= data.dueTime[i], f'C6_{cnt}_1')
            IPr.addConstr(expr >= data.readyTime[i], f'C6_{cnt}_2')
    # （7）车容量约束
    cnt = 0
    for k in range(0,data.vehicleNum):
        expr = LinExpr(0)
        for i in range(1,data.nodeNum-1):
            for j in range(0,data.nodeNum):
                if i !=j:
                    expr.addTerms(data.demand[i],X[i][j][k])
        cnt += 1
        IPr.addConstr(expr <= data.capacity, f'C7_{cnt}')
    '''
    IPr.optimize()
    for var in IPr.getVars():
        if var.X > 0:
            print(var)
    print('\n 变量列表：', IPr.getVars(), '目标函数值:', IPr.ObjVal)'''
    return IPr


''' --------------------Step 2，建立结点类，便于在分支定界中使用对应结点的相关信息-------------------- '''
class Node():
    # 定义类属性
    def __init__(self):
        self.model = None
        self.local_UB = np.inf
        self.local_LB = 0
        self.x_sol = {}  # 解是多维度的，使用字典的数据结构便于检索
        self.int_x_sol = {}
        self.is_integer = False
        self.branch_var_lst = []
        self.cnt = None

    # 定义类方法，深复制(便于在原有节点基础上，建立分枝后的子问题)
    def deepcopy(node):
        new_node = Node()  # 实例化
        new_node.model = node.model.copy()  # 复制后新结点的模型与复制前相同
        new_node.local_UB = np.inf
        new_node.local_LB = 0
        new_node.x_sol = copy.deepcopy(node.x_sol)
        new_node.int_x_sol = copy.deepcopy(node.int_x_sol)
        new_node.is_integer = node.is_integer
        new_node.branch_var_lst = []
        new_node.cnt = node.cnt
        return new_node


''' --------------------Step 3，进行分支定界------------------ '''
def BranchAndBound(IPr):
    """ ——————Step 3.1 初始化全局参数——————— """
    IPr.optimize()
    global_UB = 1000  # 由可行的初始解获得
    global_LB = IPr.ObjVal  # 对于最小化问题，可行整数解提供上界，松弛解提供下界，IPr是Gurobi的模型
    eps = 10 ** (-3)  # 0.001
    incumbent_node = Node()  # 用于覆盖存放，迄今为止所获得的最好的节点
    Gap = np.inf
    cnt = 0
    Queue = []  # 深优or广优,叶子节点集合
    Queue_IPrValue = []  # 叶子节点的松弛问题，其函数值的集合
    global_UB_lst = [global_UB]  # 记录全局上下界变化，便于出图
    global_LB_lst = [global_LB]
    # 储存cut的信息
    Cut_cnt = 0
    Cut_pool = {}
    Cut_LHS = {}

    """ ——————Step 3.2 建立初始节点—————— """
    node = Node()
    node.model = IPr.copy()
    node.local_UB = np.inf
    node.local_LB = IPr.ObjVal
    node.model.setParam('OutputFlag', 0)
    node.cnt = 0
    Queue.append(node)
    # 用于指导节点的选取规则↓
    node.model.optimize()
    Queue_IPrValue.append(node.model.ObjVal)
    """ ——————Step 3.3 分支循环—————— """
    while ((len(Queue) > 0) and (global_UB - global_LB > eps)):
        """ ——Step 3.3.1 选择队列某节点对应子问题，求解判断其整数特性—— """
        # 最小化问题，设置优先搜索松弛问题中最小的那个
        min_value = min(Queue_IPrValue)
        index = Queue_IPrValue.index(min_value)
        current_node = Queue.pop(index)  # 先入后出，深优搜索

        cnt += 1
        current_node.model.optimize()  # 求解当前节点对应的模型
        SolStatus = current_node.model.Status  # 当前子问题解的状态，在gurobi中，2-最优、3-不可行、5-无界
        is_integer = True  # 后续只要有一个分量不是整数，都会使其变为False，说明当前这个节点（问题）不是整数解
        is_Pruned = False  # 后续只要有一个符合剪枝条件，都会使其变为True，说明当前这个节点（问题）被剪枝了
        varX = []
        if SolStatus == 2:  # 对于当前节点current_node，如果求得最优解
            # 存储解的信息
            for var in current_node.model.getVars():  # 筛选出决策变量X
                split_arr = re.split(r"_", var.VarName)
                if split_arr[0] == 'X':
                    varX.append(var)
                    current_node.x_sol[var.VarName] = var.x
                    if abs(var.x - int(var.x)) > 0:  # 说明不是整数解，将这一个解放入候选分支变量集合中
                        is_integer = False
                    else:
                        current_node.int_x_sol[var.VarName] = var.x
                # if split_arr[0] == 'S':
                    # print(var.VarName, '=', var.x)

            """ ——Step 3.3.2 根据解值及其整数特性，添加cut，这里的是k-path """
            tem_cut_cnt = 0
            if is_integer == False:  # 如果不是整数解，首先进行切割再分支
                customer_set = list(range(1, data.customerNum + 1))  # 客户集合
                # print('customer_set', customer_set)
                while tem_cut_cnt < 3:  # 割平面数量
                    sample_num = random.choice(customer_set[4:])
                    selected_customer_set = random.sample(customer_set, sample_num)  # 在客户集合中，随机挑选sample_num个
                    # 对于selected_customer_set中的客户，服务这些客户的车辆数量，不少于考虑容量限制下的车辆数
                    total_demand = 0
                    for customer in selected_customer_set:
                        total_demand += data.demand[customer]
                    estimated_vehicle_num = math.ceil(total_demand/data.capacity)  # 向上取整
                    # 建立cut,针对的是变量X
                    cut_lhs = LinExpr(0)
                    for var in varX:  # 对于解中的决策变量X
                        split_arr = re.split(r"_", var.VarName)
                        if (int(split_arr[1]) not in selected_customer_set) and (int(split_arr[2]) in selected_customer_set):
                            cut_lhs.addTerms(1, var)
                    Cut_cnt += 1
                    tem_cut_cnt += 1
                    cut_name = 'Cut_' + str(Cut_cnt)
                    Cut_LHS[cut_name] = cut_lhs
                    Cut_pool[cut_name] = current_node.model.addConstr(cut_lhs >= estimated_vehicle_num, name=cut_name)
                current_node.model.update()
                # 求解添加cut之后的问题
                current_node.model.optimize()
                SolStatus = current_node.model.Status
                is_integer = True
                # cut添加完毕，进行分支
                # print('_____Cut_pool', Cut_pool)
                # print('_____Cut_LHS', Cut_LHS)
                if SolStatus == 2:  # 对于当前节点current_node，如果求得最优解
                    # 存储解的信息
                    for var in current_node.model.getVars():  # 筛选出决策变量X
                        split_arr = re.split(r"_", var.VarName)
                        if split_arr[0] == 'X':
                            current_node.x_sol[var.VarName] = copy.deepcopy(var.x)
                            if abs(var.x - int(var.x)) > 0:  # 说明不是整数解，将这一个解放入候选分支变量集合中
                                is_integer = False
                                current_node.branch_var_lst.append(var.VarName)
                            else:  # 若变量是整数
                                current_node.int_x_sol[var.VarName] = int(var.x)
                else:
                    continue

            """ ——Step 3.3.3 根据解值及其整数特性，更新局部/全局上界/下界—— """
            if is_integer == True:  # 整数解，更新全局上界，局部上下界
                current_node.is_integer = True
                current_node.local_UB = current_node.model.ObjVal
                current_node.local_LB = current_node.model.ObjVal
                if current_node.local_UB <= global_UB:
                    incumbent_node = Node.deepcopy(current_node)  # 覆盖保存当前最优节点
                global_UB = min(current_node.local_UB, global_UB)
            if is_integer == False:  # 说明有非整数分量，解不是整数解，更新局部下界（松弛解）/上界（可行解）
                current_node.is_integer = False
                current_node.local_UB = np.inf
                current_node.local_LB = current_node.model.ObjVal
            """ ——Step 3.3.4 根据条件进行剪枝—— """
            if is_integer == True:  # （1）最优性剪枝
                is_Pruned = True
            if (is_integer == False) and (current_node.local_LB >= global_UB):  # (2) 界限剪枝
                is_Pruned = True
            Gap = abs(round(100 * (global_UB - global_LB) / global_LB, 2))
            print(f" _____ {cnt} _____ Gap = {Gap}% _____ \n")
        elif (SolStatus != 2):  # 节点未得到可行解，剪枝
            is_integer = False
            is_Pruned = True
            continue

        """ ——Step 3.3.5 进行分支—— """
        if is_Pruned == False:  # 说明当前节点没有被剪枝，可以分支
            # 根据最不可行分支规则，确定分支变量，当然，分支变量也要在决策变量中确定
            branchVarName = current_node.branch_var_lst[0]
            print("branchVarName",branchVarName)
            distance_05 = abs(current_node.x_sol[branchVarName] - int(current_node.x_sol[branchVarName]) - 0.5)
            for var in current_node.branch_var_lst:
                distance_var = abs(current_node.x_sol[var] - int(current_node.x_sol[var]) - 0.5)
                if distance_var < distance_05:
                    branchVarName = var
                    distance_05 = distance_var

            # 得到分支变量的左右边界
            left_bound = int(current_node.x_sol[branchVarName])
            right_bound = int(current_node.x_sol[branchVarName]) + 1
            # 构建左右两个节点（子问题）
            left_node = Node.deepcopy(current_node)
            right_node = Node.deepcopy(current_node)
            ## 建立左结点的问题模型
            targetVar = left_node.model.getVarByName(branchVarName)
            left_node.model.addConstr(targetVar <= left_bound, name='branch left' + str(cnt))
            left_node.model.setParam('OutputFlag', 0)
            left_node.model.update()
            cnt += 1
            left_node.cnt = cnt
            ## 建立右结点的问题模型
            targetVar = right_node.model.getVarByName(branchVarName)
            right_node.model.addConstr(targetVar >= right_bound, name='branch right' + str(cnt))
            right_node.model.setParam('OutputFlag', 0)
            right_node.model.update()
            cnt += 1
            right_node.cnt = cnt
            Queue.append(left_node)
            Queue.append(right_node)
            ## 遍历叶子节点，覆盖更新下界，因为在最小化问题中，下界只能在所有叶子节点中比较
            temp_global_LB = global_UB  # 需要注意的地方！
            # Queue_checkLst = []
            # Status_checkLst = []
            # ObjVal_checkLst = []
            for node in Queue:
                node.model.optimize()
                # Queue_checkLst.append(node)
                # Status_checkLst.append(node.model.Status)
                if node.model.Status == 2:
                    # ObjVal_checkLst.append(node.model.ObjVal)
                    Queue_IPrValue.append(node.model.ObjVal)
                    if node.model.ObjVal <= temp_global_LB:
                        temp_global_LB = node.model.ObjVal
                else:
                    Queue_IPrValue.append(M)

            global_LB = temp_global_LB
            global_UB_lst.append(global_UB)
            global_LB_lst.append(global_LB)

    # 退出循环时Queue长度=0，没有输出上下界，补之
    global_UB = global_LB
    Gap = abs(round(100 * (global_UB - global_LB) / global_LB, 2))
    global_UB_lst.append(global_UB)
    global_LB_lst.append(global_LB)
    print('Queue:',Queue,len(Queue))
    print(Queue[0].model)

    print('\n\n\n')
    print('----------------------------------------------')
    print('          Branch and Bound Terminates         ')
    print('            Optimal Solution Found            ')
    print('----------------------------------------------')
    print(f'\n Final Gap = {Gap}%')
    print(f'Optimal Solution: {incumbent_node.int_x_sol}')
    print(f'Optimal Object（LB）: {global_LB}')
    print(f'Optimal Object（UB）: {global_UB}')
    return incumbent_node, Gap, global_LB_lst, global_UB_lst


''' --------------------Step 4，进行绘图-------------------- '''
def plotSolution(global_LB_lst, global_UB_lst,customerNum):
    plt.xlabel("Iteration")
    plt.ylabel("Value of Bound")
    plt.title(f"Iteration of the B&C for CVRPTW({customerNum}customers)")
    x_cor = [i + 1 for i in range(0, len(global_UB_lst))]
    plt.plot(x_cor, global_UB_lst, c='red', label='Upper bound')
    plt.plot(x_cor, global_LB_lst, c='blue', label='Lower bound')
    # plt.scatter(x_cor, global_LB_lst, s = 5, alpha=0.5, c='k')
    # plt.scatter(x_cor, global_UB_lst, s = 5, alpha=0.5, c='k')
    print(len(global_LB_lst))
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    return 0


''' --------------------Step 5，进行求解-------------------- '''
if __name__ == "__main__":
    data_path = r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\58.AP,TSP与VRPTW的分支定界、分支切割算法实现\data\C101network.txt'  # 这里是节点文件
    customerNum = 6
    vehicleNum = 3
    capacity = 50
    M = 10000
    data = readData(data_path, customerNum, vehicleNum, capacity)
    IPr = modeling(data,M)
    time_start = time.time()
    incumbent_node, Gap, global_LB_lst, global_UB_lst = BranchAndBound(IPr)
    time_end = time.time()
    print(f'计算用时：{time_end - time_start} 秒')
    plotSolution(global_LB_lst, global_UB_lst, customerNum)

