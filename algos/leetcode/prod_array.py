class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prod = 1
        zeros = 0
        for num in nums:
            if num == 0:
                zeros += 1
            prod = prod * num
        result = [0 for _ in nums]
        if zeros > 1:
            return result
        for i in range(len(result)):
            if nums[i] != 0:
                result[i] = prod//nums[i]
            else:
                special = 1
                for n in nums[:i]:
                    special = special * n
                for n in nums[i+1:]:
                    special = special * n
                result[i] = special
        return result

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        L, R, result = [0] * length, [0] * length, [0] * length
        L[0] = 1
        for i in range(1, length):
            L[i] = L[i-1] * nums[i-1]
        R[length-1] = 1
        for i in range(length-2, -1, -1):
            R[i] = R[i+1] * nums[i+1]
        for i in range(length):
            result[i] = L[i] * R[i]
        return result
