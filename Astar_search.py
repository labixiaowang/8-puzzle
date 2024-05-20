from queue import PriorityQueue  # 优先队列：在队列中按照优先级进行处理
from puzzle import Puzzle
def Astar_search(initial_state):
    start_node = Puzzle(initial_state, None, None, 0,needs_hueristic=True)
    priority_queue = PriorityQueue()  # 使用优先队列：优先级高的先处理    初始化open_list
    priority_queue.put((0, start_node))  # 评估函数初始值为0
    explored = set()  # close_list

    while not priority_queue.empty():
        _, current_node = priority_queue.get()  # 获取当前最小成本的节点，忽略第一个元素（评估函数），current_node是puzzle类的一个实例
        if current_node.goal_test():  # 判断是否满足目标状态
            return current_node.find_solution()       # 返回路径

        explored.add(tuple(current_node.state))   # 节点加入close_list
        children = current_node.generate_child()

        for child in children:   # 遍历节点的邻近节点
            if tuple(child.state) not in explored:
                # 计算A*评估函数：实际路径成本 + 启发式成本
                child.evaluation_function = child.path_cost + child.heuristic
                priority_queue.put((child.evaluation_function, child))
    return "No solution found"


