import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResnetBlock(nn.Module):
    """"
    Main block of ResNet, which contains:
      - Group Normalization
      - LeakyReLU activation
      - Convolution layers
    and then applies residual connections.
    """
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch = out_ch or in_ch

        # Sequence building main path of residual block 
        self.layers = nn.Sequential(
            # GroupNorm 
            nn.GroupNorm(num_groups=in_ch // 8, num_channels=in_ch),
            nn.LeakyReLU(),
            # 1x1 Conv 
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            # 3x3 Group Conv 
            nn.Conv2d(out_ch, out_ch, kernel_size=3, groups=4, padding=1),

            nn.GroupNorm(num_groups=out_ch // 8, num_channels=out_ch),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            # Another 3x3 Group Conv
            nn.Conv2d(out_ch, out_ch, kernel_size=3, groups=4, padding=1),
        )

        # 1x1 Conv for shortcut connection
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """
        Args
          - x: feature map [B, C, H, W]
        """
        # Add shortcut connection to main block
        return self.shortcut(x) + self.layers(x)

class AttnBlock(nn.Module):
    """
    Self-attention mechanism.
    """
    def __init__(self, ch):
        super().__init__()
        # GroupNorm 
        self.norm = nn.GroupNorm(num_groups=32, num_channels=ch)
        # 1x1 projections for Queries, Key and Values 
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        # Projection connection attention output
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the attention block.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Steps:
            1. Normalize the input tensor.
            2. Reshape to the required format for attention computation.
            3. Compute scaled dot-product attention.
            4. Reshape back and apply the output projection.
            5. Add the residual connection to the original input.
        """
        # Apply GroupNorm to normalize channels: output shape [B, C, H, W]
        h = self.norm(x)   

        # Compute query (Q), key (K), and value (V) projections
        q = self.q(h)      # [B, C, H, W]
        k = self.k(h)      # [B, C, H, W]
        v = self.v(h)      # [B, C, H, W]

        # Reshape tensors to [B, C, H*W] for attention computation
        B, C, H, W = q.shape
        q = q.reshape(B, C, H*W)
        k = k.reshape(B, C, H*W)
        v = v.reshape(B, C, H*W)

        # Compute scaled dot-product attention weights: [B, H*W, H*W]
        # Each element w[b,i,j] measures similarity between position i and j
        w = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        # Softmax over the last dimension to normalize attention scores
        w = F.softmax(w, dim=-1)  

        # Multiply attention weights by values and reshape back to [B, C, H, W]
        h = torch.einsum('bij,bcj->bci', w, v).reshape(B, C, H, W)

        # Apply final 1x1 projection and add residual connection
        h = self.proj_out(h)
        return x + h 

class Downsample(nn.Module):
    """
    Downsampling module using depthwise separable convolution.
    
    Process:
      1. Depthwise convolution with stride 2: each channel is convolved independently.
      2. Pointwise 1x1 convolution: mixes information across channels.
    """
    def __init__(self, ch, kernel_size=3, padding=1, with_conv=True):
        super().__init__()
        # Depthwise convolution: groups=ch ensures each channel is processed separately
        self.depthwise = nn.Conv2d(ch, ch, kernel_size=kernel_size, stride=2,
                                   padding=padding, groups=ch)
        # Pointwise convolution to combine channels
        self.pointwise = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Upsample(nn.Module):
    """ Upsample module using interpolation with 1x1 convolution. """
    def __init__(self, in_ch, out_ch=-1, with_conv=True):
        super().__init__()
        # 1x1 Conv
        self.conv = nn.Conv2d(in_ch, max(in_ch, out_ch), kernel_size=1) if with_conv else nn.Identity()

    def forward(self, x):
        # Increase the resolution by a factor of 2 using interpolation
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetModel(nn.Module):
    """
    U-Net architecture.
      - Downsampling path
      - Middle section with attention mechanism
      - Upsampling path
      - Skip connections between downsample and upsample layers paths
    """
    def __init__(self,
                 input_channels=3,
                 base_channels=32,
                 out_ch=1,   
                 ch_mult=(1, 2, 4),
                 num_res_blocks=2,
                 dropout=0.0,
                 resamp_with_conv=True):
        super().__init__()

        # Initial Conv2d layer with groups=1
        self.input_conv = nn.Conv2d(input_channels, base_channels, kernel_size=5, padding=2, stride=1, groups=1)

        # ---------------------
        # Downsample phase
        # ---------------------
        self.down = down = nn.ModuleList()
        in_ch = base_channels
        
        # Stores the channel numbers of each phase for skip connections
        downphase_ch = []  
        for level, mult in enumerate(ch_mult):
            out_ch_level = base_channels * mult
            downphase_ch.append(in_ch)

            # For each downsample path *num_res_blocks* ResnetBlocks are used
            for _ in range(num_res_blocks):
                down.append(
                    ResnetBlock(in_ch, out_ch_level, dropout=dropout)
                )
                downphase_ch.append(out_ch_level)
                in_ch = out_ch_level

            # Downsample at the end of each level besides last one
            if level + 1 != len(ch_mult):
                down.append(Downsample(in_ch, with_conv=resamp_with_conv))

        # ---------------------
        # Middle Section ("BottleNeck")
        # ---------------------
        self.mid = nn.ModuleList([
            ResnetBlock(in_ch, in_ch, dropout=dropout),
            AttnBlock(in_ch),
            ResnetBlock(in_ch, in_ch, dropout=dropout)
        ])

        # ---------------------
        # Upsample phase
        # ---------------------
        self.up = up = nn.ModuleList()
        for level, _ in enumerate(ch_mult):
            for _ in range(num_res_blocks + 1):

                # Take number of channels from downsample path 
                out_ch_level = downphase_ch.pop()

                # Pass through a ResNet block
                up.append(
                    ResnetBlock(in_ch + out_ch_level, out_ch_level, dropout=dropout)
                )

                # Update in_ch for the next block
                in_ch = out_ch_level

            # Upsample at the end of each level, except for the last one
            if level + 1 != len(ch_mult):
                up.append(Upsample(in_ch, with_conv=resamp_with_conv))

        # Final layer: GroupNorm -> SiLU activation -> 3x3 convolution
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: Input image of shape [B, C, H, W]
        """

        # Initial convolution 
        h = self.input_conv(x)

        # Downsample
        hs = [h]  
        for layer in self.down:
            h = layer(h)
            hs.append(h)

        # Middle section
        for layer in self.mid:
            h = layer(h)

        # Upsample
        for layer in self.up:
            # Concatenate feature maps from skip connections for ResnetBlock
            if isinstance(layer, ResnetBlock):
                # Take the feature map from top of the stack
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h)
            else:
                h = layer(h)

        return self.final(h)








    



            
    
    
    