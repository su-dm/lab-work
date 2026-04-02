"""
Given an integer n, return the least number of perfect square numbers that sum to n.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

 

Example 1:

Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
"""

# unbounded knapsack problem
class Solution:
    def numSquares(self, n: int) -> int:
        m = math.floor(math.sqrt(n))
        sqs = [(i+1)**2 for i in reversed(range(m))]
        @cache
        def dp(i, tot):
            if tot == 0:
                return 0
            if i == len(sqs):
                return float('inf')
            if sqs[i] > tot:
                return dp(i+1, tot)
            return min(dp(i, tot-sqs[i]) + 1, dp(i+1, tot))
        return dp(0, n)
    
    def numSquares(self, n: int) -> int:
        m = math.floor(math.sqrt(n))
        sqs = [(i+1)**2 for i in reversed(range(m))]

        dp = [[0] + [float('inf')]*n for _ in range(m+1)]
        # n = 0 -> 0
        for i in range(m-1, -1, -1):
            for tot in range(1, n+1):
                if tot < sqs[i]:
                    dp[i][tot] = dp[i+1][tot]
                else:
                    dp[i][tot] = min(dp[i+1][tot], dp[i][tot-sqs[i]] + 1)
        return dp[0][n]

    def numSquares(self, n: int) -> int:
        dp = [0] + [float('inf')] * n
        
        # Iterate through all totals up to n
        for tot in range(1, n + 1):
            # Try every square smaller than or equal to the current total
            j = 1
            while j * j <= tot:
                dp[tot] = min(dp[tot], dp[tot - j * j] + 1)
                j += 1
                
        return dp[n]
