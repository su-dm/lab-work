"""
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

 

Example 1:

Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
"""

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        R = len(triangle)
        @cache
        def dp(r, i):
            if r == R-1:
                return triangle[r][i]
            return min(dp(r+1, i), dp(r+1, i+1)) + triangle[r][i]
        return dp(0,0)

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        R = len(triangle)
        #dp = [[0 for _ in range(len(row))] for row in triangle]
        dp = triangle.copy()

        for r in range(R-2, -1, -1):
            for i in range(len(triangle[r])):
                dp[r][i] = min(dp[r+1][i], dp[r+1][i+1]) + triangle[r][i]
        return dp[0][0]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for r in range(1, len(triangle)):
            for i in range(len(triangle[r])):
                new_sum = min(triangle[r-1][i] if i < len(triangle[r-1]) else float('inf'), triangle[r-1][i-1] if i-1 >= 0 else float('inf')) + triangle[r][i]
                triangle[r][i] = new_sum
        return min(triangle[-1])

