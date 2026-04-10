"""
Given an n x n grid containing only values 0 and 1, where 0 represents water and 1 represents land, find a water cell such that its distance to the nearest land cell is maximized, and return the distance. If no land or water exists in the grid, return -1.

The distance used in this problem is the Manhattan distance: the distance between two cells (x0, y0) and (x1, y1) is |x0 - x1| + |y0 - y1|.

Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
Explanation: The cell (1, 1) is as far as possible from all the land with distance 2.
"""

class Solution:
    def maxDistance(self, grid: list[list[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        # Maximum possible Manhattan distance in this grid + 1
        MAX_DISTANCE = rows + cols + 1
        
        dist = [[MAX_DISTANCE] * cols for _ in range(rows)]
        
        # First pass: check left and top neighbors
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    dist[r][c] = 0
                else:
                    top = dist[r - 1][c] if r > 0 else MAX_DISTANCE
                    left = dist[r][c - 1] if c > 0 else MAX_DISTANCE
                    dist[r][c] = min(dist[r][c], min(top, left) + 1)
        
        ans = 0
        
        # Second pass: check right and bottom neighbors
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                bottom = dist[r + 1][c] if r < rows - 1 else MAX_DISTANCE
                right = dist[r][c + 1] if c < cols - 1 else MAX_DISTANCE
                dist[r][c] = min(dist[r][c], min(bottom, right) + 1)
                
                ans = max(ans, dist[r][c])
        
        # If ans is 0, there is no water. If ans is MAX_DISTANCE, there is no land.
        return ans if 0 < ans < MAX_DISTANCE else -1

    # multi source bfs, start a bfs from each land coord
    def maxDistance(self, grid: list[list[int]]) -> int:
        visited = set()
        q = deque()
        ans = 0
        R = len(grid)
        C = len(grid[0])
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    q.append((r,c,r,c))
        if len(q) == R*C:
            return -1
        ans = -1
        deltas = [(0,1),(1,0),(-1,0),(0,-1)]
        while(q):
            r,c,sr,sc = q.popleft()
            ans = max(ans, abs(r-sr)+abs(c-sc))
            for dr, dc in deltas:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in visited and grid[nr][nc] == 0:
                    q.append((nr,nc,sr,sc))
                    visited.add((nr,nc))
        return ans

    # no need to track source coord just count dist as levels
    def maxDistance(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        
        max_dist = 0
        queue = deque()
        visited = set()
        
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c]:
                    queue.append((r,c))
                    visited.add((r,c))
        
        if not queue or ROWS * COLS == len(queue):
            return -1
        
        while queue:
            for _ in range(len(queue)):
                r,c = queue.popleft()
            
                for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < ROWS and 0 <= nc < COLS and (nr,nc) not in visited \
                        and grid[nr][nc] == 0:
                            queue.append((nr,nc))
                            visited.add((nr,nc))
            max_dist += 1
        
        return max_dist - 1

