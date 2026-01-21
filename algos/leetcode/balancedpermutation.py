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

test = [4,1,3,2]
answer = "1110"

# brute force
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


def check_balance(nums):
    n = len(nums)
    pos = {}
    for i, x in enumerate(nums):
        pos[x] = i
    result = []
    min_idx = n
    max_idx = -1

    for k in range(1, n + 1):
        curr_pos = pos[k]
        if curr_pos < min_idx:
            min_idx = curr_pos
        if curr_pos > max_idx:
            max_idx = curr_pos
        # Check if the span of these numbers equals k
        if max_idx - min_idx + 1 == k:
            result.append("1")
        else:
            result.append("0")

    return "".join(result)
if __name__ == "__main__":
    result = check_balance(test)
    print(result)
    assert result == answer
