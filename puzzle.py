class Puzzle:
    goal_state=[1,2,3,8,0,4,7,6,5]   # 0表示空位
    heuristic=None   # 启发式信息值
    evaluation_function=None  # 评估启发式函数
    needs_hueristic=False #是否需要启发式函数
    num_of_instances=0  # puzzle类被创建的次数
    def __init__(self,state,parent,action,path_cost,needs_hueristic=False):
        self.parent=parent
        self.state=state
        self.action=action
        if parent:
            self.path_cost = parent.path_cost + path_cost
        else:
            self.path_cost = path_cost
        if needs_hueristic:
            self.needs_hueristic=True
            self.generate_heuristic()
            self.evaluation_function=self.heuristic+self.path_cost
        Puzzle.num_of_instances+=1

    def __str__(self):  # 打印状态，九宫格形式
        return str(self.state[0:3])+'\n'+str(self.state[3:6])+'\n'+str(self.state[6:9])

    def generate_heuristic(self):
        self.heuristic=0
        """
        实现八数码问题的启发式信息值计算，计算出来赋值给self.heuristic。
        提示：根据self.state和self.goal_state中一些值进行计算。
        """
        for i in range(len(self.state)):
            if self.state[i] != self.goal_state[i]:
                self.heuristic += 1

    def goal_test(self):
        if self.state == self.goal_state:
            return True
        return False

    @staticmethod
    def find_legal_actions(i,j):
        legal_action = ['U', 'D', 'L', 'R'] # (上，下，左，右)
        if i == 0:  # up is disable
            legal_action.remove('U')
        elif i == 2:  # down is disable
            legal_action.remove('D')
        if j == 0:
            legal_action.remove('L')
        elif j == 2:
            legal_action.remove('R')
        return legal_action

    def generate_child(self):
        children=[]
        x = self.state.index(0)  #数字0的位置
        i = int(x / 3)# 行数
        j = int(x % 3)# 列数
        legal_actions=self.find_legal_actions(i,j)

        for action in legal_actions:
            new_state = self.state.copy()  # 创建副本
            if action == 'U':# 上移
                new_state[x], new_state[x-3] = new_state[x-3], new_state[x]   # 数字0和它上面那个值交换位置
            elif action == 'D':
                new_state[x], new_state[x+3] = new_state[x+3], new_state[x]
            elif action == 'L':
                new_state[x], new_state[x-1] = new_state[x-1], new_state[x]
            elif action == 'R':
                new_state[x], new_state[x+1] = new_state[x+1], new_state[x]
            children.append(Puzzle(new_state,self,action,1,self.needs_hueristic))
        return children

    def find_solution(self):
        solution = []  # 储存解题步骤
        solution.append(self.action)
        path = self    # 创建一个变量path并将其设置为当前Puzzle对象，即解题路径的当前节点
        while path.parent != None:  # 加上父节点的路径
            path = path.parent
            solution.append(path.action)
        solution = solution[:-1]  # 去除最后一个元素
        solution.reverse() # 反转列表
        return solution

    def __lt__(self, other):
        # 这里假设evaluation_function属性包含了节点的优先级
        return self.evaluation_function < other.evaluation_function