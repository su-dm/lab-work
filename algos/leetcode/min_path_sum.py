"""
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.
"""

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])
        @lru_cache
        def dfs(r,c):
            if r >= R or c >= C:
                return float('inf')
            if r == R-1 and c == C-1:
                return grid[r][c]
            return min(dfs(r+1,c), dfs(r,c+1)) + grid[r][c]
        return dfs(0,0)
    
    def minPathSum(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])
        dp = [[float('inf')] * C for _ in range(R)]
        dp[0][0] = grid[0][0]
        for r in range(R):
            for c in range(C):
                if r == 0 and c == 0:
                    continue
                up = dp[r-1][c] if r > 0 else float('inf')
                left = dp[r][c-1] if c > 0 else float('inf')
                dp[r][c] = min(up, left) + grid[r][c]
        print(dp)
        return dp[-1][-1]
