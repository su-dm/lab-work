"""
An attendance record for a student can be represented as a string where each character signifies whether the student was absent, late, or present on that day. The record only contains the following three characters:

'A': Absent.
'L': Late.
'P': Present.
Any student is eligible for an attendance award if they meet both of the following criteria:

The student was absent ('A') for strictly fewer than 2 days total.
The student was never late ('L') for 3 or more consecutive days.
Given an integer n, return the number of possible attendance records of length n that make a student eligible for an attendance award. The answer may be very large, so return it modulo 109 + 7.

 

Example 1:

Input: n = 2
Output: 8
Explanation: There are 8 records with length 2 that are eligible for an award:
"PP", "AP", "PA", "LP", "PL", "AL", "LA", "LL"
Only "AA" is not eligible because there are 2 absences (there need to be fewer than 2).
"""

class Solution:
    def checkRecord(self, n: int) -> int:
        @cache
        def dfs(rem, absents, lates):
            if rem == 0:
                return 1
            total = 0
            if absents < 1:
                total += dfs(rem-1, absents+1, 0)
            if lates < 2:
                total += dfs(rem-1, absents, lates+1)
            total += dfs(rem-1, absents, 0)
            return total % (10**9 + 7)
        return dfs(n, 0, 0) % (10**9 + 7)

    

    def checkRecord(self, n: int) -> int:
        MOD = 1000000007
        # Initialize the cache to store sub-problem results.
        memo = [[[-1] * 3 for _ in range(2)] for _ in range(n + 1)]

        # Recursive function to return the count of combinations
        # of length 'n' eligible for the award.
        def eligible_combinations(n, total_absences, consecutive_lates):
            # If the combination has become not eligible for the award,
            # then we will not count any combinations that can be made using it.
            if total_absences >= 2 or consecutive_lates >= 3:
                return 0
            # If we have generated a combination of length 'n' we will count it.
            if n == 0:
                return 1
            # If we have already seen this sub-problem earlier,
            # we return the stored result.
            if memo[n][total_absences][consecutive_lates] != -1:
                return memo[n][total_absences][consecutive_lates]

            # We choose 'P' for the current position.
            count = eligible_combinations(n - 1, total_absences, 0)
            # We choose 'A' for the current position.
            count = (
                count + eligible_combinations(n - 1, total_absences + 1, 0)
            ) % MOD
            # We choose 'L' for the current position.
            count = (
                count
                + eligible_combinations(
                    n - 1, total_absences, consecutive_lates + 1
                )
            ) % MOD

            # Return and store the current sub-problem result in the cache.
            memo[n][total_absences][consecutive_lates] = count
            return count

        # Return count of combinations of length 'n' eligible for the award.
        return eligible_combinations(n, 0, 0)

    # python @cache decorator gets OOM but memo works
    def checkRecord(self, n: int) -> int:
        memo = [[[-1] * 3 for _ in range(2)] for _ in range(n + 1)]
        def dfs(rem, absents, lates):
            if rem == 0:
                return 1
            if memo[rem][absents][lates] != -1:
                return memo[rem][absents][lates]
            total = 0
            if absents < 1:
                total += dfs(rem-1, absents+1, 0)
            if lates < 2:
                total += dfs(rem-1, absents, lates+1)
            total += dfs(rem-1, absents, 0)
            res = total % (10**9 + 7)
            memo[rem][absents][lates] = res
            return res
        return dfs(n, 0, 0) % (10**9 + 7)
        
    def checkRecord(self, n: int) -> int:
        dp = [[[0] * 3 for _ in range(2)] for _ in range(n + 1)]
        for absent in (0, 1):
            for late in (0, 1, 2):
                dp[0][absent][late] = 1
        
        for r in range(1, n+1):
            for absent in (0, 1):
                for late in (0, 1, 2):
                    count = 0
                    if late < 2:
                        count += dp[r-1][absent][late+1]
                    if absent < 1:
                        count += dp[r-1][absent+1][0]
                    count += dp[r-1][absent][0]
                    dp[r][absent][late] = count % (10**9 + 7)
        return dp[n][0][0]
            
