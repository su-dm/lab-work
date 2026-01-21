"""
Given a permutation $p$ of length $n$, a number $k$ is balanced if there are two indices l, r
such that the numbers $[p[l], p[l+1],... p[r]] form a permutation of the numbers 1, 2, ..., k.
For each k (1 <= k <= n), determine if it is balanced. 
Return a binary string of length n where the i^{th} character is '1' if i is balanced and '0' otherwise.
A permutation of length n contains each integer from 1 to n exactly once in any order.
n = 4
Example
p = [4, 1, 3, 2]
For k = 1: 
Choose l = 2, r = 2, so p[2:2] = [1], which is a permutation of length 1.
For k = 2: No pair of indices results in a permutation of length 2, so it is not balanced.
For k = 3: Choose l = 2, r = 4, so p[2:4] = [1, 3, 2], which is a permutation of length 3.
For k = 4: Choose l = 1, r = 4, so p[1:4] = [4, 1, 3, 2], which is a permutation of length 4.
"""

import ipdb
test = [4,1,3,2]
answer = "1110"




#[3,1,4,2,2,1,8,9,10]

# brute force
def check_balance(nums):
    ipdb.set_trace()
    def explore(idx):
        num = nums[idx] - 1
        # total length is nums[idx], if you're testing 'balanced' 4 then you need 1,2,3,4 max length 4
        # so if we've already found our starting/main number, you need nums[idx] - 1 or num more elements in the list
        # that's why max_length (addtional) is num
        max_length = num
        l,r= idx, idx
        while(num > 0):
            found = False
            for i in range(idx, max(0, idx-max_length),-1):
                if r - i + 1 > nums[idx]:
                    return '0'
                # within range
                if nums[i] == num:
                    l = min(l, i)
                    found = True
                    break
            # we use far left idx to check left side first, if first number isn't found we use farthest right index as max
            # resetting initial conditions
            if not found:
                for i in range(idx, min(len(nums), idx+num+1)):
                #search on right
                    if i - l + 1 > nums[idx]:
                        return '0'
                    # within range
                    if nums[i] == num:
                        r = max(r, i)
                        found = True
                        break
            if found:
                num -= 1
            else:
                return '0'
        return '1'

    res = ""
    for i in range(len(nums)):
        # checking if num at index is balanced
        res += explore(i)
    return res

test = [4,1,3,2]
def check_balance(nums):
    def check(idx):
        #ipdb.set_trace()
        num = nums[idx]
        target_set = {i for i in range(1, num+1)}
        window_len = num
        l_min = max(0, idx-(window_len-1))
        r_max = min(len(nums)-1, idx + (window_len-1))
        l_max = r_max - (window_len-1)
        for i in range(l_min, l_max + 1):
            if set(nums[i:i+window_len]) == target_set:
                return '1'
        return '0'
    result = ""
    for i in range(len(nums)):
        valid = check(i)
        result += valid 
    return result



if __name__ == "__main__":
    result = check_balance(test)
    print(result)
    assert result == answer











#def check_balance(nums, k):
#    l,r = 0
#    cur = 0
#    n = len(nums)
#    def dfs(idx):
#        num = nums[idx]
#        # if num - 1 within num-1 distance (left or right) then it's ok
#        # make that the left or right
#        # go to next number with the updated left and rights
#        # but if there's duplicate numbers you could have solutions on both sides... have to keep queue to explore multiple
#    while(cur < n):
#        num = nums[cur]
#
