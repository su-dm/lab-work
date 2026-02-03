"""
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.
"""

class Solution:
    # recursive top down
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse=True)
        @lru_cache
        def explore(idx, rem):
            if rem == 0:
                return 0
            if idx >= len(coins):
                return float('inf')
            i = rem // coins[idx]
            ans = float('inf')
            while(i >= 0):
                poss = explore(idx + 1, rem - (i * coins[idx])) + i
                if poss is not float('inf'):
                    ans = min(ans, poss)
                i -= 1
            return ans
        result = explore(0, amount)
        return result if result != float('inf') else -1

    # maxsize=None means potential OOM
    # maxsize=128 is the default
    # https://stackoverflow.com/questions/61536704/why-does-python-lru-cache-performs-best-when-maxsize-is-a-power-of-two
    # improve memoization by reducing parameters/dimensions, more call stacks / depth but caching trumps it
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse=True)
        @lru_cache
        def explore(rem):
            if rem == 0:
                return 0
            ans = float('inf')
            for coin in coins:
                if rem - coin >= 0:
                    poss = explore(rem-coin) + 1
                    ans = min(ans, poss)
            return ans
        result = explore(amount)
        return result if result != float('inf') else -1

    # bottom up dp
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                if coin > amount:
                    continue
                rem = i - coin
                if rem >= 0:
                    dp[i] = min(dp[i], dp[rem]+1)
        return dp[amount] if dp[amount] != float('inf') else -1

    # optimize to remove unecessary check
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1 
