import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class ART_block(nn.Module):
    """
    ART_block（Attention Residual Transformer Block）
    功能描述：
    结合 CNN 的局部特征提取与 Transformer 的全局注意力机制的复合模块。
    通过下采样将特征图送入 Transformer 提取全局依赖，再将输出上采样后与残差 CNN 特征进行拼接融合。

    【参数说明】:
        - config: Transformer 和相关结构的超参数配置字典/对象。
        - input_dim: 输入特征的维度大小。
        - img_size: 输入图像/特征图的空间尺寸 (H, W)。
        - transformer: 可选的 Transformer 实例，若提供则启用全局注意力分支。
    """
    def __init__(self,config, input_dim, img_size,transformer = None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        self.config = config
        ngf = 64  # 基础通道数目
        mult = 4  # 基础倍率
        use_bias = False
        norm_layer = nn.BatchNorm2d  # 对于batch数小可以改成InstanceNorm
        padding_type = 'reflect'
        if self.transformer:
            # Downsample，channel_size ngf*4 ——》 1024
            model = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3,
                               stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.Conv2d(ngf * 8, 1024, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(1024),
                      nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))
            #Patch embedings
            self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)
            # Upsampling block
            model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 8,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.ReLU(True)]
            setattr(self, 'upsample', nn.Sequential(*model))
            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        """
        前向传播逻辑：

        【输入 `x`】: 
            Tensor, 形状为 (B, C, H, W)，其中 C 是当前特征图通道数。
            
        【输出】:
            Tensor, 形状与输入几乎保持一致（由于经过残差CNN与Transformer融合），具体维度取决于各子模块的具体设计。
            通常为 (B, C_out, H, W)。
        """
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x)
            # embed
            embedding_output = self.embeddings(down_sampled)
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output)
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            transformer_out = transformer_out.permute(0, 2, 1)
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w)
            # upsample transformer output
            transformer_out = self.upsample(transformer_out)
            # concat transformer output and resnet output
            x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x

    class Embeddings(nn.Module):
        """Construct the embeddings from patch, position embeddings.
        """

        def __init__(self, config, img_size, in_channels=3, input_dim=3, old=1):
            super(Embeddings, self).__init__()
            self.config = config
            img_size = _pair(img_size)
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            in_channels = 1024
            # Learnable patch embeddings
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
            # learnable positional encodings
            self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
            self.dropout = Dropout(config.transformer["dropout_rate"])

        def forward(self, x):
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            embeddings = x + self.positional_encoding
            embeddings = self.dropout(embeddings)
            return embeddings


class Embeddings(nn.Module):
    """
    Embeddings 模块
    功能描述：
    为 Transformer 模型构建 Patch Embeddings 和 Position Embeddings。
    将 2D 的特征图通过卷积切分为独立的 Patch 并展平为 1D 序列，同时加入位置编码以保留空间信息。
    """
    def __init__(self, config, img_size, in_channels=3,input_dim=3,old = 1):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        in_channels = 1024
        #Learnable patch embeddings
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        """
        前向传播逻辑：

        【输入 `x`】:
            Tensor, 形状通常为 (B, in_channels, H, W) (这是下采样后的特征图，例如通道数为1024)。
            
        【输出 `embeddings`】:
            Tensor, 形状为 (B, n_patches, hidden_size)。
            这些是由 2D 图转为 1D 序列并加入位置编码后的特征，可直接输入到 Transformer 中。
        """
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings


