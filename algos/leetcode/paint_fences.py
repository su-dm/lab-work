class Solution:
    def numWays(self, n: int, k: int) -> int:
        @lru_cache
        def dp(i):
            if i == 0:
                return k
            if i == 1:
                return k*k
            # color differently theres k-1 option in each perm
            diff = dp(i-1) * (k-1)
            # color same is dp(i-1) but cant repeat 3
            # so how many ways can i-1 be diff than i-2?
            same = dp(i-2) * (k-1)
            return diff + same
        return dp(n-1)

	def numWays(self, n: int, k: int) -> int:
        def total_ways(i):
            if i == 1:
                return k
            if i == 2:
                return k * k
            
            # Check if we have already calculated totalWays(i)
            if i in memo:
                return memo[i]
            
            # Use the recurrence relation to calculate total_ways(i)
            memo[i] = (k - 1) * (total_ways(i - 1) + total_ways(i - 2))
            return memo[i]

        memo = {}
        return total_ways(n)

	def numWays(self, n: int, k: int) -> int:
        if n < 3:
            import math
            return int(math.pow(k,n))
        dp = [0] * n
        dp[0] = k
        dp[1] = k*k
        for i in range(2, n):
            dp[i] = (k-1) * (dp[i-1] + dp[i-2])
        return dp[n-1]
