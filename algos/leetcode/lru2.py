class Node:
    def __init__(self, key=None, val=None):
        self.prev = None
        self.next = None
        self.key = key
        self.val = val

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity
        self.size = 0
        # dummy nodes
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.items = {}

    def _remove(self, item):
        item.prev.next =  item.next
        item.next.prev = item.prev
        del self.items[item.key]

    def _add(self, item):
        self.head.next.prev = item
        item.next = self.head.next
        item.prev = self.head
        self.head.next = item
        self.items[item.key] = item

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.items:
            return -1
        
        item = self.items[key]
        self._remove(item)
        self._add(item)
        return item.val
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        
        if key in self.items:
            self._remove(self.items[key])
        item = Node(key, value)
        if len(self.items) == self.cap:
            self._remove(self.tail.prev)
        self._add(item)
        

