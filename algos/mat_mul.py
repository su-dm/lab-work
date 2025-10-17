def mat_mul(a, b):
    assert len(a[0]) == len(b) # (m x n) x (n x p) = (m x p)
    M, N, P = len(a), len(b), len(b[0])
    result = []
    for m in range(M):
        row = []
        for p in range(P):
            pairs = zip(a[m], [b[n][p] for n in range(N)])
            mult = [x[0] * x[1] for x in pairs]
            val = sum(mult)
            row.append(val)
        result.append(row)
    return result
