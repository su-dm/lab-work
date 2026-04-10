"""
You are given several boxes with different colors represented by different positive numbers.

You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (i.e., composed of k boxes, k >= 1), remove them and get k * k points.

Return the maximum points you can get.

 

Example 1:

Input: boxes = [1,3,2,2,2,3,4,3,1]
Output: 23
Explanation:
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
----> [1, 3, 3, 4, 3, 1] (3*3=9 points) 
----> [1, 3, 3, 3, 1] (1*1=1 points) 
----> [1, 1] (3*3=9 points) 
----> [] (2*2=4 points)
"""

"The actual DP N^4 solution is pretty confusing"

class Solution:
	# stupid brute force method
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        #@cache
        def dfs(taken, rem):
            if len(rem) == 0:
                return 0
            ans = 0
            for idx in rem:
                new_taken = taken.copy()
                new_rem = rem.copy()
                new_taken.add(idx)
                new_rem.remove(idx)
                removed = 1
                # check left
                left_idx = idx
                right_idx = idx
                for i in range(idx-1, -1, -1):
                    if i in new_taken:
                        continue
                    elif boxes[i] == boxes[idx]:
                        new_taken.add(i)
                        new_rem.remove(i)
                        removed += 1
                    else:
                        break
                for i in range(idx+1, N):
                    if i in new_taken:
                        continue
                    elif boxes[i] == boxes[idx]:
                        new_taken.add(i)
                        new_rem.remove(i)
                        removed += 1
                    else:
                        break
                ans = max(ans, dfs(new_taken, new_rem) + removed**2)
            return ans
        return dfs(set(), set(range(N)))

	# bit better
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        @cache
        def dfs(rem):
            if len(rem) == 0:
                return 0
            ans = 0
            i = 0
            while(i < len(rem)):
                j = i + 1
                while(j < len(rem) and boxes[rem[j]] == boxes[rem[i]]):
                    j += 1
                removed = j - i
                new_rem = rem[:i] + rem[j:]
                ans = max(ans, dfs(new_rem) + removed**2)
                i = j
            return ans
        return dfs(tuple(range(N)))
