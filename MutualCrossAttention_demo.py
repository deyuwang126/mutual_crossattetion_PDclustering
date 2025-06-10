# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:13:50 2025

@author: Movement Rehab Lab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# --- Mutual Cross-Attention Module Definition ---
# This class implements a bidirectional cross-attention mechanism.
# Unlike standard Transformer cross-attention, it does NOT use linear projections
# (W_Q, W_K, W_V) to transform input features. It assumes inputs are already
# in a compatible feature space.
class CrossAttention(nn.Module): 
    def __init__(self, dropout_rate): 
        super(CrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) # Use the provided dropout rate

    def forward(self, x1, x2):

        d = x1.shape[-1] # Get the feature dimension, assuming x1 and x2 have the same last dimension

        # Attention from x1 to x2 (x1 as Query, x2 as Key/Value)
        # scores = Q @ K.T / sqrt(d_k)
        scores_x1_to_x2 = torch.bmm(x1, x2.transpose(1, 2)) / math.sqrt(d)
        # attention_output = softmax(scores) @ V
        output_from_x1_to_x2 = torch.bmm(self.dropout(F.softmax(scores_x1_to_x2, dim=-1)), x2)

        # Attention from x2 to x1 (x2 as Query, x1 as Key/Value)
        scores_x2_to_x1 = torch.bmm(x2, x1.transpose(1, 2)) / math.sqrt(d)
        output_from_x2_to_x1 = torch.bmm(self.dropout(F.softmax(scores_x2_to_x1, dim=-1)), x1)

        # Summation of the two intermediate outputs
        output = output_from_x1_to_x2 + output_from_x2_to_x1

        return output
    
    
# --- Data Loading and Preparation --- 
print("Loading 'toy_dataset1.npy' and 'toy_dataset2.npy'...")   
loaded_toy_dataset1_np = np.load('toy_dataset1.npy')
loaded_toy_dataset2_np = np.load('toy_dataset2.npy')

# Convert NumPy arrays to PyTorch tensors
# .float() ensures the data type is torch.float32, which is standard for neural networks.
input_feature_A_2d = torch.from_numpy(loaded_toy_dataset1_np).float()
input_feature_B_2d = torch.from_numpy(loaded_toy_dataset2_np).float()

# Add the batch dimension (batch_size = 1) for the attention module
input_feature_A_3d = input_feature_A_2d.unsqueeze(0) # Shape: (1, 40, 24)
input_feature_B_3d = input_feature_B_2d.unsqueeze(0) # Shape: (1, 40, 24)

# Instantiate and use the Mutual CrossAttention
cross_attn_modified = CrossAttention(dropout_rate=0)
output_modified = cross_attn_modified(input_feature_A_3d, input_feature_B_3d)
print(f"Output shape from Modified CrossAttention: {output_modified.shape}") # Expected: (1, 40, 24)



# --- Post-Processing for Clustering and Visualization ---
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(output_modified.squeeze(0).detach().cpu().numpy())

from sklearn.manifold import TSNE


# 使用 t-SNE 将特征降到 2D
tsne = TSNE(n_components=2, random_state=42)
fused_features_2d = tsne.fit_transform(output_modified.squeeze(0).detach().cpu().numpy())
import matplotlib.pyplot as plt
import seaborn as sns
# --- Plotting ---
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    x=fused_features_2d[:, 0],
    y=fused_features_2d[:, 1],
    hue=cluster_labels, # Color points by their cluster assignments
    palette='viridis', # Choose a color palette
    legend='full',
    s=80,
    alpha=0.8
)
plt.title('t-SNE Visualization of Attention Output with K-Means Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()