"""
Given two strings s and t, return the number of distinct subsequences of s which equals t.

The test cases are generated so that the answer fits on a 32-bit signed integer.

 

Example 1:

Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from s.
rabbbit
rabbbit
rabbbit
"""

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        @cache
        def dp(start_s, idx_t):
            if idx_t == len(t):
                return 1
            if start_s == len(s):
                return 0
            num = 0
            for i in range(start_s, len(s)):
                if s[i] == t[idx_t]:
                    num += dp(i+1, idx_t+1)
            return num
        return dp(0, 0)

    def numDistinct(self, s: str, t: str) -> int:
        S = len(s)
        T = len(t)
        dp = [[0]*T+[1] for _ in range(S+1)]
        for i_s in range(S-1, -1, -1):
            for i_t in range(T-1, -1, -1):
                if s[i_s] == t[i_t]:
                    dp[i_s][i_t] = dp[i_s+1][i_t+1] + dp[i_s+1][i_t]
                else:
                    dp[i_s][i_t] = dp[i_s+1][i_t]
        return dp[0][0]
