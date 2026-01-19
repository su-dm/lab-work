def all_combinations(nums):
    combos = []
    for i in range(len(nums)):
        # without copy it iterates on newly appended values
        for combo in combos.copy():
            combos.append(combo + [nums[i]])
        combos.append([nums[i]])
    return combos + [[]]

def all_combinations(nums):
    combos = [[]]
    for num in nums:
        combos += [combo + [num] for combo in combos]
    return combos

# recursive
def all_combinations(nums):
    if not nums:
        return [[]]
    rest_combos = all_combinations(nums[1:])
    return rest_combos + [combo + [nums[0]] for combo in rest_combos]

def test_combinations():
    test1 = [1,2,3]
    answer = [[], [1], [1,2], [2], [1,3], [1,2,3], [2,3], [3]]
    result = all_combinations(test1)
    print(f"All combinations for test1:{test1} result:{result}")
    result_set = {tuple(sorted(combo)) for combo in result}
    answer_set = {tuple(combo) for combo in answer}
    assert result_set == answer_set

if __name__ == "__main__":
    test_combinations()
