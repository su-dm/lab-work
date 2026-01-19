def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	# (m x n) -> (n x m)
    new = [[ for _ in range(len(a))] for _ in range(len(a[0]))]
    for m in range(len(a)):
        for n in range(len(a[0])):
            new[n][m] = a[m][n]
    return new

def transpose2(a):
    rows = len(a)
    cols = len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]

def transpose3(a):
	return [list(row) for row in zip(*a)]
