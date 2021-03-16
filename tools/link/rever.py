
class Node:
    def __init__(self, val, next):
        self.val = val
        self.next = next

def reverNodes(node):
    pre = node
    cur = node.next
    pre.next = None

    while cur:
        temp = cur.next
        cur.next = pre
        pre = cur
        cur = temp
    return pre


if __name__ == "__main__":
    node1 = Node(1, None)
    node2 = Node(2, None)
    node3 = Node(3, None)
    node4 = Node(4, None)

    node1.next = node2
    node2.next = node3
    node3.next = node4

    node1 = reverNodes(node1)
    temp = node1
    while temp:
        print(temp.val)
        temp = temp.next
