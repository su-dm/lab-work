"""
    You want to schedule a list of jobs in d days. Jobs are dependent (i.e To work on the ith job, you have to finish all the jobs j where 0 <= j < i).

You have to finish at least one task every day. The difficulty of a job schedule is the sum of difficulties of each day of the d days. The difficulty of a day is the maximum difficulty of a job done on that day.

You are given an integer array jobDifficulty and an integer d. The difficulty of the ith job is jobDifficulty[i].

Return the minimum difficulty of a job schedule. If you cannot find a schedule for the jobs return -1.
    Input: jobDifficulty = [6,5,4,3,2,1], d = 2
Output: 7
Explanation: First day you can finish the first 5 jobs, total difficulty = 6.
Second day you can finish the last job, total difficulty = 1.
The difficulty of the schedule = 6 + 1 = 7 
"""

def minDifficulty(self, jobDifficulty: List[int], days: int) -> int:
    # d days, i index of next job to schedule
    # at each thing you have option of for i..n is a day
    # top down is dfs(i,d)
    # bottom up you gotta think about the base, start from 1 day left
    N = len(jobDifficulty)
    if days > N:
        return -1
    
    # what's inner and outer loop, i vs d
    # d outer cause base case is when you only have 1 day left
    dp = [[float('inf')]*(N+1) for _ in range(days+1)]
    # if there's 0 days left and you're out of bounds on jobs you know you're done 0 diff
    dp[0][N] = 0
    for d in range(1, days+1):
        # you have to take at least 1 job, and if you're taking 1 the latest you can schedule is (d-1) before the last job cause you still need at least d-1 jobs to schedule
        for start_job in range(N - d + 1):
            cur_diff = jobDifficulty[start_job]
            # you can can schedule anything between those
            for end_job in range(start_job, N - d + 1):
                cur_diff = max(cur_diff, jobDifficulty[end_job])
                dp[d][start_job] = min(dp[d][start_job], dp[d-1][end_job+1] + cur_diff)
    return dp[days][0]
