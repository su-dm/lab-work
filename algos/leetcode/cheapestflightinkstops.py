class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist = {i:float('inf') for i in range(n)}
        adj = defaultdict(list)
        for flight in flights:
            adj[flight[0]].append((flight[1], flight[2]))
        
        q = deque()
        q.append(([src], 0))
        stops = 0

        while stops <= k and q:
            level_size = len(q)
            for _ in range(level_size):
                path, path_cost = q.popleft()
                cur_loc = path[-1]
                for dest,cost in adj[cur_loc]:
                    new_path_cost = cost + path_cost
                    if new_path_cost < dist[dest]:
                        dist[dest] = new_path_cost
                        q.append((path + [dest], new_path_cost))
            stops += 1
        if dist[dst] != float('inf'):
            return dist[dst]
        else:
            return -1
