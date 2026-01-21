"""
    You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.

 

Example 1:


Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
Example 2:

Input: times = [[1,2,1]], n = 2, k = 1
Output: 1
Example 3:

Input: times = [[1,2,1]], n = 2, k = 2
Output: -1
"""

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        costs = {i:float('inf') for i in range(1, n+1)}
        adj = defaultdict(set)
        for s,d,c in times:
            adj[s].add((d,c))
        q = deque()
        start = ([k], 0)
        costs[k] = 0
        q.append(start)
        while(q):
            path, p_cost = q.popleft()
            for nei, cost in adj[path[-1]]:
                if nei not in path:
                    new_cost = p_cost + cost
                    if new_cost < costs[nei]:
                        costs[nei] = new_cost
                        q.append((path + [nei], new_cost))

        print(costs)
        mx = max(costs.values())
        if mx == float('inf'):
            return -1
        else:
            return mx

# improve solution with priority queue
class Solution:
    from collections import defaultdict
    import heapq
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        adj = defaultdict(list)
        visited = set()
        weights = {}
        for time in times:
            u,v,w = time
            adj[u].append(v)
            weights[(u,v)] = w
        # start from k ensure n reached
        # use dist to track min dist to that node
        dist = {}
        dist[k] = 0
        q = [(0, k)]
        while(q):
            cost, cur = heapq.heappop(q)
            if cost > dist[cur]:
                continue
            for hop in adj[cur]:
                next_cost = weights[(cur,hop)] + cost
                if hop not in dist or next_cost < dist[hop]:
                    dist[hop] = next_cost
                    heapq.heappush(q, (next_cost, hop))
        if len(dist) != n:
            return -1
        return max(dist.values())
