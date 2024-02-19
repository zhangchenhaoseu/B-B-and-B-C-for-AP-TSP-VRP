# 靡不有初，鲜克有终
# zhangchenhaoseu@foxmail.com
# Southeast University
# 2024/1/30 21:27

# branch and bound for numerical case
# 所给的 （IP）问题 如下
# min -(100x_0+150x_1)
# 2x_0 + x_1 <= 10
# 3x_0 + 6x_1 <= 40
# x_0,x_1 >=0, and integer

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import time
import copy


''' --------------------Step 1，使用guroubi建立（IP）的线性松弛问题（IPr）-------------------- '''
def modeling():
    IPr = Model("IPr")  # Step 1.1 建立模型
    X = [[] for _ in range(2)]  # Step 1.2 建立变量
    for i in range(0, len(X)):
        X[i] = IPr.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"X_{i}")
    IPr.update()
    obj = LinExpr(0)  # Step 1.3 建立目标函数
    obj_coeffs = [-100, -150]
    obj.addTerms(obj_coeffs, X)
    IPr.setObjective(obj, sense=GRB.MINIMIZE)
    expr1, expr2 = LinExpr(0), LinExpr(0)  # Step 1.4 建立约束条件
    cons1_coeffs, cons2_coeffs = [2, 1], [3, 6]
    expr1.addTerms(cons1_coeffs, X)
    expr2.addTerms(cons2_coeffs, X)
    IPr.addConstr(expr1 <= 10, 'C1')
    IPr.addConstr(expr2 <= 40, 'C2')
    # IPr.optimize()
    IPr.write('IPr_NumericalCase.lp')
    # print('变量列表：', IPr.getVars(), '目标函数值:', IPr.ObjVal)
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
    global_UB = 0  # 由可行的初始解获得
    global_LB = IPr.ObjVal   # 对于最小化问题，可行整数解提供上界，松弛解提供下界，IPr是Gurobi的模型
    eps = 10**(-3)  # 0.001
    incumbent_node = Node  # 用于覆盖存放，迄今为止所获得的最好的节点
    Gap = np.inf
    cnt = 0
    Queue = []  # 深优or广优,叶子节点集合
    global_UB_lst = [global_UB]  # 记录全局上下界变化，便于出图
    global_LB_lst = [global_LB]

    """ ——————Step 3.2 建立初始节点—————— """
    node = Node()
    node.model = IPr.copy()
    node.local_UB = np.inf
    node.local_LB = IPr.ObjVal
    node.model.setParam('OutputFlag', 0)
    node.cnt = 0
    Queue.append(node)

    """ ——————Step 3.3 分支循环—————— """
    while ((len(Queue) > 0) and (global_UB-global_LB > eps)):
        """ ——Step 3.3.1 选择队列某节点对应子问题，求解判断其整数特性—— """
        current_node = Queue.pop()  # 先入后出，深优搜索
        cnt += 1
        current_node.model.optimize()  # 求解当前节点对应的模型
        SolStatus = current_node.model.Status  # 当前子问题解的状态，在gurobi中，2-最优、3-不可行、5-无界
        is_integer = True  # 后续只要有一个分量不是整数，都会使其变为False，说明当前这个节点（问题）不是整数解
        is_Pruned = False  # 后续只要有一个符合剪枝条件，都会使其变为True，说明当前这个节点（问题）被剪枝了
        if SolStatus == 2:  # 对于当前节点current_node，如果求得最优解
            # 存储解的信息
            for var in current_node.model.getVars():  # 对于解中的任意一个变量
                print(var.VarName, '=', var.x)
                current_node.x_sol[var.VarName] = var.x
                if abs(var.x - int(var.x)) > eps:  # 说明不是整数解，将这一个解放入候选分支变量集合中
                    is_integer = False
                    current_node.branch_var_lst.append(var.VarName)
                else:  # 若变量是整数
                    current_node.int_x_sol[var.VarName] = int(var.x)
            """ ——Step 3.3.2 根据解值及其整数特性，更新局部/全局上界/下界—— """
            if is_integer == True:  # 整数解，更新全局上界，局部上下界
                current_node.is_integer = True
                current_node.local_UB = current_node.model.ObjVal
                current_node.local_LB = current_node.model.ObjVal
                if current_node.local_UB < global_UB:
                    incumbent_node = Node.deepcopy(current_node)  # 覆盖保存当前最优节点
                global_UB = min(current_node.local_UB, global_UB)
            elif is_integer == False:  # 说明有非整数分量，解不是整数解，更新局部下界（松弛解）/上界（可行解）
                current_node.is_integer = False
                current_node.local_UB = np.inf
                current_node.local_LB = current_node.model.ObjVal
            """ ——Step 3.3.3 根据条件进行剪枝—— """
            if is_integer == True:  # （1）最优性剪枝
                is_Pruned = True
            if (is_integer == False) and (current_node.local_LB > global_UB):  # (2) 界限剪枝
                is_Pruned = True
            Gap = abs(round(100*(global_UB - global_LB)/global_LB,2))
            print(f" _____ {cnt} _____ Gap = {Gap}% _____ \n")
        else:  # 节点未得到可行解，剪枝
            is_integer = False
            is_Pruned = True

        """ ——Step 3.3.4 进行分支—— """
        if is_Pruned == False:  # 说明当前节点没有被剪枝，可以分支
            # 根据最不可行分支规则，确定分支变量
            branchVarName = current_node.branch_var_lst[0]
            distance_05 = abs(current_node.x_sol[branchVarName] - int(current_node.x_sol[branchVarName])-0.5)
            for var in current_node.branch_var_lst:
                distance_var = abs(current_node.x_sol[var] - int(current_node.x_sol[var])-0.5)
                if distance_var < distance_05:
                    branchVarName = var
                    distance_05 = distance_var
            # 得到分支变量的左右边界
            left_bound = int(current_node.x_sol[branchVarName])
            right_bound = left_bound + 1
            # 构建左右两个节点（子问题）
            left_node = Node.deepcopy(current_node)
            right_node = Node.deepcopy(current_node)
            ## 建立左结点的问题模型
            targetVar = left_node.model.getVarByName(branchVarName)
            expr_left = targetVar <= left_bound
            left_node.model.addConstr(expr_left, name='branch left' + str(cnt))
            left_node.model.setParam('OutputFlag',0)
            left_node.model.update()
            cnt += 1
            left_node.cnt = cnt
            ## 建立右结点的问题模型
            targetVar = right_node.model.getVarByName(branchVarName)
            expr_right = targetVar >= right_bound
            right_node.model.addConstr(expr_right, name='branch right' + str(cnt))
            right_node.model.setParam('OutputFlag', 0)
            right_node.model.update()
            cnt += 1
            right_node.cnt = cnt
            Queue.append(left_node)
            Queue.append(right_node)
            ## 遍历叶子节点，覆盖更新下界，因为在最小化问题中，下界只能在所有叶子节点中比较
            temp_global_LB = global_UB
            for node in Queue:
                node.model.optimize()
                if node.model.status==2:
                    if node.model.ObjVal <= temp_global_LB:
                        temp_global_LB = node.model.ObjVal
            global_LB = temp_global_LB
            global_UB_lst.append(global_UB)
            global_LB_lst.append(global_LB)

    # 退出循环时Queue长度=0，没有输出上下界，补之
    global_UB = global_LB
    Gap = abs(round(100*(global_UB - global_LB)/global_LB,2))
    global_UB_lst.append(global_UB)
    global_LB_lst.append(global_LB)

    print('\n\n')
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
def plotSolution(global_LB_lst, global_UB_lst):
    plt.xlabel("Iteration")
    plt.ylabel("Value of Bound")
    plt.title("Iteration of the B&B for Numerical Case")
    x_cor = [i+1 for i in range(0, len(global_UB_lst))]
    plt.plot(x_cor, global_UB_lst, c='red', label='Upper bound')
    plt.plot(x_cor, global_LB_lst, c='blue', label='Lower bound')
    # plt.scatter(x_cor, global_LB_lst, alpha=1, c='k')
    # plt.scatter(x_cor, global_UB_lst, alpha=1, c='k')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    return 0


''' --------------------Step 5，进行求解-------------------- '''
if __name__ =="__main__":
    IPr = modeling()
    time_start = time.time()
    incumbent_node, Gap, global_LB_lst, global_UB_lst = BranchAndBound(IPr)
    time_end = time.time()
    print(f'计算用时：{time_end-time_start} 秒')
    plotSolution(global_LB_lst, global_UB_lst)


