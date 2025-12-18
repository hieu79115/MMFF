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
    def __init__(self, in_channels=3, num_class=60, dataset='ntu', edge_importance_weighting=True, **kwargs):
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
            ST_GCN_Block(in_channels, 64, kernel_size, 1, residual=False),
            ST_GCN_Block(64, 64, kernel_size, 1),
            ST_GCN_Block(64, 64, kernel_size, 1),
            ST_GCN_Block(64, 128, kernel_size, 2), # Stride 2 để giảm chiều thời gian
            ST_GCN_Block(128, 128, kernel_size, 1),
            ST_GCN_Block(128, 128, kernel_size, 1),
            ST_GCN_Block(128, 256, kernel_size, 2),
            ST_GCN_Block(256, 256, kernel_size, 1),
            ST_GCN_Block(256, 256, kernel_size, 1),
        ))

        # Output feature dim (để nối với RGB stream)
        self.out_dim = 256

    def forward(self, x):
        # Input x shape: (N, C, T, V, M) - Batch, Channel, Time, Vertex(Joint), Person
        # Chúng ta xử lý gộp Person (M) vào Batch hoặc lấy trung bình.
        # Giả sử input dataset đưa vào là (N, C, T, V) (1 người)
        
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        # Chạy qua các lớp ST-GCN
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # x lúc này có shape (N, 256, T, V) -> Đây là Feature Map cần cho Attention
        feature_map = x 

        # Global Pooling (Spatial + Temporal) cho Late Fusion
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1) # Vector (N, 256)
        
        # TRẢ VỀ CẢ 2
        return x, feature_map