from dgcnn import DGCNN
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch
import torch.nn as nn

''''
MAIN IDEA

DGCNN features => GIN => correspondence matrix [B, N, N] => pc2_matched => C @ pc2 => Estimate (R, t) ← pc1, pc2_matched

'''

# GIN architecture
class GINEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super.__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,hidden_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,hidden_channels)
        ))

        def forward(self,data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x,edge_index)
            x = self.conv2(x,edge_index)
            return x 

class ProposedModel(nn.Module):
    def __init__(self, dgcnn, gin_hidden_dim=64, k=20):
        self.dgcnn = dgcnn
        self.k = k
        self.gin = GINEncoder(in_channels=dgcnn.args.emb_dims, hidden_channels=gin_hidden_dim)

    def build_graph(self, features):
        '''
        Computes graph edges to the nearest k points and converts to PyG Format
        '''
        edge_index = knn_graph(features,k=self.k,loop=False)
        return Data(x=features, edge_index=edge_index)
    
    def forward(self, pc1, pc2):
        """
        pc1, pc2: [B, D, N]
        Returns:
            soft correspondence matrices: [B, N, N]
        """
        B, _, N = pc1.shape
        soft_correspondece = []

        for b in range(B):
            f1 = self.dgcnn.forward_features(pc1[b].unsqueeze(0))[0] # [N.D]
            f2 = self.dgcnn.forward_features(pc2[b].unsqueeze(0))[0] # [N,D]

            # build PyG graphs
            g1 = self.build_graph(f1)
            g2 = self.build_graph(f2)

            # GIN feature extraction
            z1 = self.gin(g1) # [N, hidden]
            z2 = self.gin(g2) # [N, hidden]

            # compute similarity and soft correspondence
            sim = torch.matmul(z1,z2.T)
            prob = F.softmax(sim,dim=1) #[N,N]

            soft_correspondece.append(prob)

        return torch.stack(soft_correspondece) # [B,N,N]
    
def find_rigid_alignment_batch(A, B):
    """
    Batched Kabsch algorithm for rigid alignment.

    Args:
        A: [B, N, 3] - source points
        B: [B, N, 3] - target points (matched)

    Returns:
        R: [B, 3, 3] - rotation matrices
        t: [B, 3, 1] - translation vectors
    """
    B_size, N, _ = A.shape

    centroid_A = A.mean(dim=1, keepdim=True)  # [B, 1, 3]
    centroid_B = B.mean(dim=1, keepdim=True)

    A_centered = A - centroid_A  # [B, N, 3]
    B_centered = B - centroid_B

    H = torch.matmul(A_centered.transpose(1, 2), B_centered)  # [B, 3, 3]

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-2, -1)
    R = torch.matmul(V, U.transpose(-2, -1))  # [B, 3, 3]

    # Fix reflection (ensure det(R) = +1)
    det = torch.det(R).unsqueeze(-1).unsqueeze(-1)
    V[:, :, -1] *= torch.sign(det).squeeze(-1)
    R = torch.matmul(V, U.transpose(-2, -1))

    t = centroid_B.transpose(1, 2) - torch.matmul(R, centroid_A.transpose(1, 2))  # [B, 3, 1]
    return R, t


def align_pointclouds_from_correspondence(pc1,pc2,corr):
    """
    Estimate rigid transformation from soft correspondence matrix.

    Args:
        pc1: [B, 3, N] - source point cloud
        pc2: [B, 3, N] - target point cloud
        corr: [B, N, N] - soft correspondence matrix (pc1 -> pc2)

    Returns:
        R: [B, 3, 3] - rotation matrices
        t: [B, 3, 1] - translation vectors
        aligned_pc1: [B, 3, N] - transformed source clouds
    """
    B, _, N = pc1.shape

    pc1_t = pc1.transpose(1, 2).contiguous()  # [B, N, 3]
    pc2_t = pc2.transpose(1, 2).contiguous()  # [B, N, 3]

    pc2_matched = torch.bmm(corr, pc2_t)  # [B, N, 3], pc2_matched = C @ pc2

    R, t = find_rigid_alignment_batch(pc1_t, pc2_matched)

    aligned_pc1 = torch.matmul(R, pc1) + t

    return R, t, aligned_pc1