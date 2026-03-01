import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import Graph

class ConvTemporalGraphical(nn.Module):
    # Eq 15: Phép nhân Graph Convolution
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 
                              kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), 
                              dilation=(1, 1), bias=True)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # Thực hiện phép nhân với ma trận kề A (Eq 195)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), ((kernel_size[0] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = lambda x: x

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)

class SkeletonStream_STGCN(nn.Module):
    def __init__(self, in_channels=3, num_class=60, dataset='ntu', edge_importance_weighting=True, dropout: float = 0.0, **kwargs):
        super().__init__()
        # Load Graph NTU-RGBD
        self.graph = Graph(dataset=dataset)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Cấu hình các lớp ST-GCN (Spatial: Kernel 1, Temporal: Kernel 9)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        # Build networks
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        # Các layer ST-GCN nối tiếp nhau
        self.st_gcn_networks = nn.ModuleList((
            ST_GCN_Block(in_channels, 64, kernel_size, 1, dropout=dropout, residual=False),
            ST_GCN_Block(64, 64, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(64, 64, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(64, 128, kernel_size, 2, dropout=dropout), # Stride 2 để giảm chiều thời gian
            ST_GCN_Block(128, 128, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(128, 128, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(128, 256, kernel_size, 2, dropout=dropout),
            ST_GCN_Block(256, 256, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(256, 256, kernel_size, 1, dropout=dropout),
            ST_GCN_Block(256, 256, kernel_size, 1, dropout=dropout),  # New 6th layer at 256 channels
        ))

        # Edge Importance Weighting (ST-GCN paper): learnable mask for adjacency A per layer.
        self.edge_importance_weighting = bool(edge_importance_weighting)
        if self.edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.A)) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = None

        # Output feature dim (để nối với RGB stream)
        self.out_dim = 256

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) for single-person  OR  (N, C, T, V, M) for multi-person.
        Returns:
            vec:         (N, 256)           — global feature vector
            feature_map: (N, 256, T', V)    — spatial-temporal feature map for cross-attention
        """
        if x.dim() == 5:
            # Multi-person: (N, C, T, V, M) → gộp M vào batch
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        else:
            N, C, T, V = x.size()
            M = 1

        # BatchNorm trên input
        x = x.permute(0, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous().view(N * M, C, T, V)

        # Chạy qua các lớp ST-GCN
        if self.edge_importance_weighting and self.edge_importance is not None:
            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                x = gcn(x, self.A * importance)
        else:
            for gcn in self.st_gcn_networks:
                x = gcn(x, self.A)

        # x: (N*M, 256, T', V)
        feature_map = x

        # Global Pooling (Spatial + Temporal)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N * M, -1)  # (N*M, 256)

        if M > 1:
            # Pool across persons: (N*M, 256) → (N, M, 256) → mean → (N, 256)
            x = x.view(N, M, -1).mean(dim=1)
            # Feature map: (N*M, 256, T', V) → (N, M, 256, T', V) → mean → (N, 256, T', V)
            _, Cout, Tout, Vout = feature_map.size()
            feature_map = feature_map.view(N, M, Cout, Tout, Vout).mean(dim=1)

        return x, feature_map