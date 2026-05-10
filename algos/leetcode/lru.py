class Node:
    def __init__(self, key, val):
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
        self.head = None
        self.tail = None
        self.items = {}

    def move_to_head(self, item):
        if item.prev is not None:
            if item.next is None:
                self.tail = item.prev
            item.prev.next = item.next
            if item.next is not None:
                item.next.prev = item.prev
            item.next = self.head
            item.prev = None
            self.head.prev = item
            self.head = item

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.items:
            return -1
        item = self.items[key]
        self.move_to_head(item)
        return item.val
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.items:
            item = self.items[key]
            item.val = value
            self.move_to_head(item)
            return
        item = Node(key, value)
        if len(self.items) == self.cap:
            # drop tail
            dropped = self.tail
            if self.cap == 1:
                self.head = None
                self.tail = None
            else:
                self.tail = dropped.prev
                self.tail.next = None
            del self.items[dropped.key]
        if self.head is None:
            self.head = item
            self.tail = item
        else:
            item.next = self.head
            self.head.prev = item
            self.head = item
        
        self.items[key] = item
