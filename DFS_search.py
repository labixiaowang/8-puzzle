from queue import Queue
from puzzle import Puzzle
class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def is_empty(self):
        return not bool(self.items)
    def size(self):
        return len(self.items)


def depth_first_search(initial_state):
    start_node = Puzzle(initial_state, None, None, 0)
    stack = Stack()  # 使用栈代替队列
    stack.push(start_node)
    explored = []

    while not stack.is_empty():
        node = stack.pop()  # 从栈顶取出节点
        if node.goal_test():
            return node.find_solution()
        explored.append(node.state)
        children = node.generate_child()

        for child in children:
            if child.state not in explored:
                stack.push(child)  # 将子节点压入栈中
    return

