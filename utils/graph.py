import numpy as np

class Graph:
    def __init__(self, strategy='spatial', dataset='ntu'): # Thêm tham số dataset
        self.dataset = dataset
        self.get_edge()
        self.hop_size = 1
        self.get_adjacency(strategy)

    def get_edge(self):
        if self.dataset == 'ntu':
            # Cấu hình 25 khớp của NTU RGB+D (Giữ nguyên code cũ)
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            self.edge = [(i - 1, j - 1) for (i, j) in neighbor_link] + self_link
            
        elif self.dataset == 'utd':
            # Cấu hình 20 khớp của UTD-MHAD (Kinect v1)
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            # Sơ đồ nối khớp của Kinect v1
            neighbor_link = [
                (1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                (1, 17), (17, 18), (18, 19), (19, 20)
            ]
            # Chuyển index từ 1-based sang 0-based
            self.edge = [(i - 1, j - 1) for (i, j) in neighbor_link] + self_link
        else:
            raise ValueError("Dataset không hỗ trợ")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, i] == 0: # self
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, i] > self.hop_dis[i, i]: # further
                                a_further[j, i] = normalize_adjacency[j, i]
                            else: # closer
                                a_close[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_root + a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Chưa hỗ trợ strategy này")

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
        
    @property
    def hop_dis(self):
        # Tính khoảng cách giữa các node (thuật toán đơn giản)
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis