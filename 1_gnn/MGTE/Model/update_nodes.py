def update(xs, A, M, i):
    """
    Update the node vectors in a graph
    considering their neighboring node vectors (i.e., sum or mean),
    which are non-linear transformed by neural network."""
    u_xs = xs  # embedding过后的fingerprint，比如shape为7x16
    u_A = A  # 大的邻接矩阵，比如shape为7x7
    u_M = M  # 用于求平均值的系数，比如[4,4,4,4,3,3,3]
    u_hs = torch.selu(W_fingerprint[i](u_xs))  # 对fingerprint进行非线性变换
    if update_method == 'sum':
        return u_xs + torch.matmul(u_A, u_hs)  # 然后和邻接矩阵点乘，取sum或者平均
    if update_method == 'mean':
        return u_xs + torch.matmul(u_A, u_hs) / (u_M - 1)

A = [[0., 6., 0., 6., 0., 0., 0.],
        [6., 7., 7., 3., 0., 0., 0.],
        [1., 3., 2., 0., 0., 0., 0.],
        [1., 3., 0., 6., 0., 0., 0.],
        [0., 0., 0., 0., 1., 6., 7.],
        [0., 0., 0., 0., 2., 8., 9.],
        [0., 0., 0., 0., 2., 2., 4.]]