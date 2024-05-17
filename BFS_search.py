from queue import Queue
from puzzle import Puzzle


def breadth_first_search(initial_state):
    start_node = Puzzle(initial_state, None, None, 0)
    q = Queue()  #FOFI
    q.put(start_node)
    explored=[]  # 记录已经探索过的状态
    while not(q.empty()):
        node=q.get()
        if node.goal_test():
            return node.find_solution()
        explored.append(node.state)
        children=node.generate_child()  # 生成所有可能的子节点
        for child in children:
            if child.state not in explored:
                q.put(child)
    return
