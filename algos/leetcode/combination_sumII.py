"""
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

 

Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
 

Constraints:

1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30
"""

# no negative targets or candidates by problem definition
class Solution:
    # memory limit exceeded
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        nums = sorted(candidates, reverse=True)
        combos = [[]]
        for i in range(len(nums)):
            if nums[i] > target:
                continue
            combos += [combo + [nums[i]] for combo in combos if sum(combo) + nums[i] <= target]
        deduped = set([tuple(sorted(combo)) for combo in combos if sum(combo) == target])
        return [list(combo) for combo in deduped]

    # problem allows for duplicate candidates
    # optimize by tracking duplicates and sum while building combos
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        result = set()
        nums = sorted(candidates, reverse=True)
        combos = {((),target)}
        for i in range(len(nums)):
            if nums[i] > target:
                continue
            new_combos = set()
            for combo in combos:
                c, tot = combo
                if c + (nums[i],) in combos:
                    continue
                new_tot = tot - nums[i]
                if new_tot == 0:
                    valid_combo = tuple(sorted(c + (nums[i],)))
                    result.add(valid_combo)
                elif new_tot > 0:
                    new_combos.add((c+(nums[i],),new_tot))
            combos.update(new_combos)
        return [list(combo) for combo in result]

    # recursive backtrack solution
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        nums = sorted(candidates)
        result = []
        
        def backtrack(combo, total, start_idx):
            if total == target:
                result.append(combo[:])
                return
            for i in range(start_idx, len(nums)):
                if total + nums[i] > target:
                    break
                if i > start_idx and nums[i] == nums[i-1]:
                    continue
                combo.append(nums[i])
                backtrack(combo, total + nums[i], i+1)
                combo.pop()
        backtrack([], 0, 0)
        return result


