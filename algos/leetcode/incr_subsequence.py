"""
Given an integer array nums, return the length of the longest strictly increasing subsequence.

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
"""

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        @cache
        def dp(start):
            cur_max = 1
            for j in range(start+1, len(nums)):
                if nums[j] > nums[start]:
                    cur_max = max(cur_max, dp(j) + 1)
            return cur_max
        
        return max(dp(i) for i in range(len(nums)))

    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for start in range(len(nums) - 1, -1, -1):
            cur_max = 1
            for j in range(start + 1, len(nums)):
                if nums[j] > nums[start]:
                    cur_max = max(cur_max, dp[j] + 1)
            dp[start] = cur_max
        return max(dp)
