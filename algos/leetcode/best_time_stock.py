# you can only buy once
def maxProfit(self, prices: List[int]) -> int:
    cur_min = prices[0]
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] < cur_min:
            cur_min = prices[i]
        else:
            max_profit = max(max_profit, prices[i] - cur_min)
    return max_profit

# buy as many times are you want, you can only hold 1 stock
def maxProfit(self, prices: List[int]) -> int:
    # vars are the day/idx and whether i'm holding
    # if holding you can sell or do nothing, if not you can buy or do nothing
    @lru_cache
    def dp(i, holding):
        # last day sell whatcha got
        if i == len(prices) - 1:
            if holding:
                return prices[-1]
            else:
                return 0

        profit_action = 0
        if holding:
            # sell
            profit_action = prices[i] + dp(i+1, False)
        else:
            # buy
            profit_action = -prices[i] + dp(i+1, True)
        profit_no_action = dp(i+1, holding)
        return max(profit_action, profit_no_action)
    return dp(0, False)
            

# same now bottom up
def maxProfit(self, prices: List[int]) -> int:
    dp = [[0,0] for _ in range(len(prices)+1)]
    for i in range(len(prices)-1,-1,-1):
        dp[i][0] = max(dp[i+1][0], dp[i+1][1] - prices[i])
        dp[i][1] = max(dp[i+1][1], dp[i+1][0] + prices[i])
    # can't hold on day 0
    return dp[0][0]

# peak/valley approach, sum any positive diffs
def maxProfit(self, prices: List[int]) -> int:
    maxprofit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            maxprofit += prices[i] - prices[i - 1]
    return maxprofit

# k transactions allowed, but you can only hold 1, this is buy and sell IV
def maxProfit(self, k: int, prices: List[int]) -> int:
    @cache
    def dp(i, k_rem, holding):
        # last day sell whatcha got
        if i == len(prices) - 1:
            if holding:
                return prices[-1]
            else:
                return 0

        profit_sell = 0
        profit_buy = 0
        profit_nothing = 0
        if holding:
            profit_sell = prices[i] + dp(i+1, k_rem, False)
        elif k_rem:
            profit_buy = -prices[i] + dp(i+1, k_rem-1, True)
        profit_nothing = dp(i+1, k_rem, holding)
        return max([profit_buy, profit_sell, profit_nothing])
    return dp(0, k, False)

def maxProfit(self, k: int, prices: List[int]) -> int:
    dp = [ [[0,0] for _ in range(k+1)] for _ in range(len(prices)+1)]
    for i in range(len(prices)-1,-1,-1):
        for k_rem in range(k+1):
            dp[i][k_rem][0] = max(dp[i+1][k_rem][0], dp[i+1][k_rem-1][1] - prices[i] if k_rem else 0)
            dp[i][k_rem][1] = max(dp[i+1][k_rem][1], dp[i+1][k_rem][0] + prices[i])
    # can't hold on day 0
    return dp[0][k][0]
