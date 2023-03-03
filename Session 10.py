#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.patch_embed(x) # shape: [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = rearrange(x, 'b e h w -> b (h w) e') # flatten patches into sequence
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, self.patch_embedding(x)), dim=1)
        x = x + self.positional_embedding
        
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # average over sequence length dimension
        x = self.fc(x)
        return x


# In[2]:


import torch
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.fc_q = torch.nn.Linear(input_dim, input_dim)
        self.fc_k = torch.nn.Linear(input_dim, input_dim)
        self.fc_v = torch.nn.Linear(input_dim, input_dim)
        self.fc_o = torch.nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Split input into num_heads and compute Q, K, V
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values and combine
        attention_outputs = torch.matmul(attention_weights, v)
        attention_outputs = attention_outputs.permute(0, 2, 1, 3).contiguous()
        attention_outputs = attention_outputs.view(batch_size, seq_len, input_dim)
        
        # Project back to input dimension and add residual connection
        attention_outputs = self.fc_o(attention_outputs)
        attention_outputs = attention_outputs + x
        
        return attention_outputs


# In[3]:


import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.fc_q = torch.nn.Linear(input_dim, input_dim)
        self.fc_k = torch.nn.Linear(input_dim, input_dim)
        self.fc_v = torch.nn.Linear(input_dim, input_dim)
        self.fc_o = torch.nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Split input into num_heads and compute Q, K, V
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values and combine
        attention_outputs = torch.matmul(attention_weights, v)
        attention_outputs = attention_outputs.permute(0, 2, 1, 3).contiguous()
        attention_outputs = attention_outputs.view(batch_size, seq_len, input_dim)
        
        # Project back to input dimension and add residual connection
        attention_outputs = self.fc_o(attention_outputs)
        attention_outputs = attention_outputs + x
        
        return attention_outputs


# In[4]:


import numpy as np

# assume we have a flattened patch with shape (9, 512)
flattened_patch = np.random.rand(9, 512)

# split each vector into 64 dimensions
split_patch = np.split(flattened_patch, 8, axis=1)

# check the shape of the split patch
print(split_patch[0].shape) # should output (9, 64)


# In[5]:


import numpy as np

# assume we have 8 patches with shape (9, 64)
patches = [np.random.rand(9, 64) for _ in range(8)]

# assume we have a query vector with shape (1, 64)
query = np.random.rand(1, 64)

# calculate the self-attention for each patch separately
for i, patch in enumerate(patches):
    # calculate the key and value vectors for the patch
    key = patch @ W_k[i] + b_k[i] # assume W_k and b_k are learned parameters
    value = patch @ W_v[i] + b_v[i] # assume W_v and b_v are learned parameters
    
    # calculate the dot product similarity between the query and each key vector
    similarity = (query @ key.T) / np.sqrt(key.shape[1])
    
    # calculate the attention weights using softmax
    weights = softmax(similarity)
    
    # calculate the weighted sum of the value vectors
    weighted_sum = weights @ value
    
    # concatenate the outputs for each patch
    outputs.append(weighted_sum)
    
# concatenate the outputs for all patches
output = np.concatenate(outputs, axis=1)


# In[ ]:


import numpy as np

# Generate some random query and document vectors
num_queries = 10
num_documents = 20
embedding_size = 100
queries = np.random.randn(num_queries, embedding_size)
documents = np.random.randn(num_documents, embedding_size)

# Calculate cosine similarities using matrix multiplication
cosine_similarities = queries @ documents.T / (np.linalg.norm(queries, axis=1)[:, np.newaxis] @ np.linalg.norm(documents, axis=1)[np.newaxis, :])

# Print the resulting matrix of cosine similarities
print(cosine_similarities)


# In[1]:


import torch
import torch.nn.functional as F

# Define the number of heads and the embedding size
num_heads = 8
embedding_size = 64

# Generate some random input data
batch_size = 4
seq_length = 9
input_data = torch.randn(batch_size, seq_length, embedding_size)

# Define the linear projections for each head
W_q = torch.randn(embedding_size, embedding_size // num_heads)
W_k = torch.randn(embedding_size, embedding_size // num_heads)
W_v = torch.randn(embedding_size, embedding_size // num_heads)

# Apply the linear projections to the input data to obtain the queries, keys, and values for each head
queries = input_data @ W_q
keys = input_data @ W_k
values = input_data @ W_v

# Split the queries, keys, and values into multiple heads along the embedding dimension
queries = queries.view(batch_size, seq_length, num_heads, embedding_size // num_heads).permute(0, 2, 1, 3)
keys = keys.view(batch_size, seq_length, num_heads, embedding_size // num_heads).permute(0, 2, 1, 3)
values = values.view(batch_size, seq_length, num_heads, embedding_size // num_heads).permute(0, 2, 1, 3)

# Compute the dot products between the queries and keys for each head
dot_products = torch.matmul(queries, keys.transpose(-2, -1))

# Apply softmax to the dot products to obtain attention weights for each head
attention_weights = F.softmax(dot_products, dim=-1)

# Compute the weighted sum of the values using the attention weights for each head
weighted_sum = torch.matmul(attention_weights, values)

# Concatenate the outputs of the individual heads and multiply by another weight matrix
output = weighted_sum.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, embedding_size)
W_o = torch.randn(embedding_size, embedding_size)
final_output = output @ W_o

# Print the final output tensor
print(final_output)


# In[ ]:


import torch
import torch.nn as nn

# Define the dimensions of the input and output vectors
input_size = 512
hidden_size = 2048
output_size = 512

# Generate some random input data
batch_size = 4
seq_length = 9
input_data = torch.randn(batch_size, seq_length, input_size)

# Define the feedforward neural network
ffn = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# Apply the feedforward neural network to the input data
output_data = ffn(input_data)

# Print the output tensor
print(output_data)


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.utils import load_state_dict_from_url

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # assuming 3-channel RGB images
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        
        # create patch embeddings
        self.patch_embeddings = nn.Linear(patch_dim, dim)
        
        # add position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, dim))
        
        # add [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # create MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        # split image into patches
        x = rearrange(x, 'b c h w -> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)
        
        # flatten patches and add [CLS] token
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), rearrange(x, 'b h w c -> b (h w) c')], dim=1)
        
        # apply patch embeddings and position embeddings
        x = self.patch_embeddings(x)
        x += self.position_embeddings
        
        # apply transformer encoder
        x = self.transformer_encoder(x)
        
        # extract [CLS] token representation
        cls_token = x[:, 0]
        
        # apply MLP head for classification
        logits = self.mlp_head(cls_token)
        return logits


# In[2]:


import torch

def split_image_into_patches(image, patch_size):
    # image: PyTorch tensor of shape (batch_size, channels, height, width)
    # patch_size: integer representing the size of the square patches
    
    batch_size, channels, height, width = image.size()
    assert height % patch_size == 0 and width % patch_size == 0, "image size must be divisible by patch size"
    
    num_patches = (height // patch_size) * (width // patch_size)
    patch_height = patch_width = patch_size
    
    # reshape image into patches
    patches = image.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    patches = patches.contiguous().view(batch_size, channels, num_patches, patch_height, patch_width)
    
    return patches


# In[3]:


import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten()
        )

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_patches, emb_size)
        
        x = self.proj(x) # apply convolutional and linear layers
        num_patches = x.size(1) # determine number of patches
        return x.transpose(1, 2) # swap second and third dimensions


# In[4]:


import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(),
            nn.Linear(emb_size * (patch_size ** 2), emb_size)
        )

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_patches, emb_size)
        
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(-1, x.size(1), self.patch_size, self.patch_size)
        embeddings = self.proj(patches)
        embeddings = embeddings.view(-1, self.num_patches, embeddings.size(-1))
        return embeddings


# In[5]:


import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_patches, emb_size)
        
        x = self.proj(x) # apply convolutional layer
        x = x.flatten(start_dim=2) # flatten height and width dimensions
        x = x.transpose(1, 2) # swap second and third dimensions
        return x


# In[6]:


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_size, num_classes, num_layers, hidden_size, dropout):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(p=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=8,
                dim_feedforward=hidden_size,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_classes)
        
        x = self.patch_embed(x) # convert image patches to embeddings
        batch_size, num_patches, emb_size = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # repeat [CLS] token along batch dimension
        x = torch.cat((cls_tokens, x), dim=1) # prepend [CLS] token to embeddings
        x += self.pos_embed # add positional embeddings
        x = self.dropout(x) # apply dropout to embeddings
        x = x.transpose(0, 1) # swap batch and sequence dimensions
        x = self.transformer(x) # apply transformer encoder to embeddings
        x = x.mean(dim=0) # average embeddings across sequence dimension to get a single embedding per batch
        x = self.fc(x) # apply linear layer to final embeddings
        return x


# In[7]:


import math

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.patch_embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches+1, emb_size))

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_patches+1, emb_size)
        
        x = self.patch_embedding(x) # apply convolutional and linear layers
        x = x.flatten(2).transpose(1, 2) # reshape to (batch_size, num_patches, emb_size)
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # prepend cls token to patch embeddings
        
        x = x + self.positional_embeddings # add positional embeddings
        
        return x


# In[8]:


import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size, max_seq_len):
        super().__init__()
        self.emb_size = emb_size
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, emb_size))

    def forward(self, x):
        batch_size, num_patches, emb_size = x.shape
        # Add position embedding to the [CLS] token
        cls_token = torch.zeros(batch_size, 1, emb_size).to(x.device)
        x = torch.cat([cls_token, x], dim=1)
        # Add positional embeddings to patches and [CLS] token
        x = x + self.pos_embedding[:num_patches+1]
        return x
patch_size = 16
in_channels = 3
emb_size = 768
max_seq_len = (image_size // patch_size) ** 2 + 1

# Split image into patches
patches = patch_embed(image)

# Get [CLS] token
cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

# Add [CLS] token to patches
patches = torch.cat([cls_token, patches], dim=1)

# Add positional embeddings to patches and [CLS] token
pos_embedding = PositionalEmbedding(emb_size, max_seq_len)
combined_embed = pos_embedding(patches)


# In[9]:


import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        
        mlp_hidden_dim = int(emb_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, emb_size),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(emb_size)
        
    def forward(self, x):
        # x: PyTorch tensor of shape (seq_len, batch_size, emb_size)
        # output: PyTorch tensor of shape (seq_len, batch_size, emb_size)
        
        # self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # feed-forward network
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x


# In[11]:


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, emb_size=768, num_heads=12, 
                 num_layers=12, mlp_ratio=4.0):
        super().__init__()
        
        assert (img_size % patch_size) == 0, "image size must be divisible by patch size"
        
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, emb_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, emb_size))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(emb_size, num_classes)
        
    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, channels, height, width)
        # output: PyTorch tensor of shape (batch_size, num_classes)
        
        # patch embedding
        x = self.patch_embedding(x)
        batch_size, num_patches, emb_size = x.size()
        
        # prepend cls token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # add positional embedding
        x = x + self.positional_embedding
        
        # transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # final classification
        x = x.mean(dim=1)
     


# In[12]:


import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, feedforward_dim, dropout_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emb_size, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, emb_size)
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: PyTorch tensor of shape (seq_len, batch_size, emb_size)
        # output: PyTorch tensor of shape (seq_len, batch_size, emb_size)
        
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = residual + x

        return x


# In[13]:


B, N, C = combined_embedding.shape
combined_embedding = combined_embedding.view(B, -1, C).transpose(0, 1)  # reshape and transpose


# In[14]:


class MLPBlock(nn.Module):
    def __init__(self, emb_size, mlp_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, mlp_hidden_size)
        self.fc2 = nn.Linear(mlp_hidden_size, emb_size)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, seq_len, emb_size)
        # output: PyTorch tensor of shape (batch_size, seq_len, emb_size)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_hidden_size, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(emb_size, num_heads, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.mlp_block = MLPBlock(emb_size, mlp_hidden_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: PyTorch tensor of shape (batch_size, seq_len, emb_size)
        # output: PyTorch tensor of shape (batch_size, seq_len, emb_size)

        attention_output = self.self_attention(x)
        x = self.layer_norm1(x + self.dropout(attention_output))
        mlp_output = self.mlp_block(x)
        x = self.layer_norm2(x + self.dropout(mlp_output))
        return x


# In[15]:


return self.mlp_head(out[:, 0])


# In[16]:


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)[:, 0]  # use first token (CLS token) as the image representation
        x = self.head(x)

        return x


# In[ ]:




