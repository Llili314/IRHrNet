from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import cv2   #Ҫ��imshow
import torch
import numpy as np
import os
import math
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import Adam
import torchvision.models as models
from utils import *

###�����timesformerģ�飬����Ƕ����MFA���༶�������ɣ�

# PreNormģ���Ŀ������ÿ��ע�������ǰ�������֮ǰӦ�ò��һ������
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# ǰ������㣨FeedForward����Transformerģ����������Ҫ�����ã�����������ǿģ�͵ķ����Ա������������ת������
# ǰ�������ͨ��������ȫ���Ӳ���ɣ���ṹ������ʾ
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention
# ���ڼ���ע����Ȩ�ز���������м�Ȩ���
def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention_up(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # Split out classification token at index 0
        q_ = q
        k_ = k
        v_ = v
        # cls_q, q_, = q[:, 0:1], q[:, 1:]
        # cls_k, k_, = k[:, 0:1], k[:, 1:]
        # cls_v, v_, = v[:, 0:1], v[:, 1:]

        # Rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # Attention
        out = attn(q_, k_, v_)

        # Merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # Merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Apply linear transformation
        out = self.to_out(out)

        return out

## ��׶��������ɿ飬CNL ͨ�����ɣ�PNL �ռ伯��
class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(MFA_block, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Linear(self.low_dim, self.low_dim)
        self.theta = nn.Linear(self.high_dim, self.low_dim)
        if flag == 0:
            self.phi = nn.Linear(self.low_dim, self.low_dim)
            self.W = nn.Sequential(nn.Linear(self.low_dim, self.high_dim),
                                   nn.BatchNorm1d(high_dim), )
        else:
            self.phi = nn.Linear(self.low_dim, self.low_dim)
            self.W = nn.Sequential(nn.Linear(self.low_dim, self.high_dim),
                                   nn.BatchNorm1d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B, N, D = x_h.shape
        x_h = rearrange(x_h, 'b n d -> (b n) d')
        x_l = rearrange(x_l, 'b n d -> (b n) d')
        #output_data = []
        #for b in range(x_h.shape[0]):
        # slice_x_h = x_h[b, :]
        # slice_x_l = x_l[b, :]
        g_x = self.g(x_l)
        theta_x = self.theta(x_h)
        phi_x = self.phi(x_l)
        energy = torch.matmul(theta_x, phi_x.t())
        attention = energy / energy.size(-1)
        y = torch.matmul(attention, g_x)
        W_y = self.W(y)
        z = W_y + x_h
        z = rearrange(z, '(b n) d -> b n d', b=B, n=N)
        #output_data.append(z)
        #output_data = torch.stack(output_data, dim=0)
        return z

# main classes
class TimeSformer_update(nn.Module):
    def __init__(
            self,
            *,
            dim=224,  # ����ά��
            num_frames=20,
            num_classes=2, # �ò���
            image_size=224,
            patch_size=16,
            channels=3,
            depth=4, # network depth
            heads=8,
            dim_head=64,
            attn_dropout=0.1,   # �����ɵ�
            ff_dropout=0.1      # �����ɵ�
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)  # ��TimeSformerģ���е�һ������ͶӰ�㣬���ڽ�������Ƶ��ÿһ֡����Ϊ�飬����ÿ����ӳ�䵽Ƕ��ռ��С�
        # ����to_patch_embedding����һ�����Ա任��nn.Linear�����������������ά�ȣ�patch_dim��ת��Ϊģ���е�����ά�ȣ�dim����������Ա任����ѧϰ��ԭʼ����ӳ�䵽Ƕ��ռ��е�Ȩ�ء�
        #self.pos_emb = nn.Embedding(num_positions + 1, dim)
        # pos_emb��������Ϊ���������е�ÿ��λ�����λ�ñ��룬������λ����Ϣ����ͨ��Ƕ��㽫λ������ӳ��Ϊλ��������������Щλ�������ӵ������Ƕ���ʾ�У�ʹ��ģ�Ϳ��Ի�ù���ÿ��λ�õ�λ����Ϣ���Ӷ����õؽ�ģ�������ݡ�

        #self.cls_token = nn.Parameter(torch.randn(1, dim))

        # ʹ��nn.Parameter�����Խ��������Ϊģ�͵Ŀ�ѧϰ������������ģ�͵�ѵ���������Զ������ݶȼ���Ͳ�������
        # nn.Parameter(torch.randn(1, dim))������һ����״Ϊ(1, dim)���������������װΪһ����ѧϰ�Ĳ�����
        # ���������ֵ������ģ�͵�ѵ�������и�����ʧ�������ݶ��½��㷨�����Ż�
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_up(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  # Time attention
                PreNorm(dim, Attention_up(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  # Spatial attention
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))  # Feed Forward
            ]))
        # ���ڶ����x����rPPG�ź�
        self.upsample = nn.Sequential(
            # nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            # nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )
        self.ConvBlock1 = nn.Conv1d(dim, dim // 2, 1, stride=1, padding=0)
        self.ConvBlock2 = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)
        #�����ã�ע�͵�
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.MFA = MFA_block(224, 224, 0)  # ��ʼ��MFA_block, ��Ϊ����ͼ��768 = 3*16*16������ά�����ǰ����3
    def forward(self, video):
        #b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        b, f, _, *_, device = *video.shape, video.device
        #assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        #n = (h // p) * (w // p) # ������Ϊ196���飬224/16 * 224/16
        n = 1 #��������ֿ飬����n=1��Ϊ����������rearrange
        # video 13(b),588(3*14*14),768(16*16*3)
        #video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        # tokens 13, 588, 224, ��768��ά��224������
        tokens = self.to_patch_embedding(video)
        # cla tokenά�� 13,1,224
        #cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        # x->13,589,224
        #x = torch.cat((cls_token, tokens), dim=1)
        # x-> 13,588,224
        x = tokens
        # ��λ��Ƕ�������Ӻ͵���������x��ÿ��λ���ϣ�������λ����Ϣ�����λ�ù�ϵ x->13,589,224
        #x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x_t = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            # ʱ��ע���������������x����ʱ��ע��������
            x_s = spatial_attn(x_t, 'b (f n) d', '(b f) n d', f=f) + x_t
            x_ = ff(x_s) + x_s
            x_f = self.MFA(x_, x)  # �������ɣ�����ֱ�Ϊ��ʱע�������Լ�ǰ����������q��k��v�Ĺ�ϵ���м��ɣ������ƶ�
            x = x_f  # ����x

        # 13,224
        # �������ַ�cls��Ӧ�������Ϊencoder��������� ���������յ�image presentation����һ�������ǲ���cls�ַ��������е�tokens�������һ��ƽ�������ڷ���
        # ���ﲻ��cls�������xΪB,588,224������rPPG�źŵĹ���
        #cls_token = x[:, 0]  # ��cls����clsֱ����Ϊ���
        # cls_token = torch.mean(x_f, dim=1)  # ����cls��������tokens��ƽ����Ϊ���
        # cls_token = self.to_out(cls_token)
        #����rPPG�ź�
        x_f = x_f.transpose(1, 2)  # [B, 224, f, 14, 14]
        #x_f = self.upsample(x_f)  # [B,224,3,14,14]
        #x_f = self.upsample2(x_f) # [B,112,3,14,14]
        #x_f = torch.mean(x_f, 3)  # [B,112,3,14]
        #x_f = torch.mean(x_f, 3)  # [B,112,3]
        x_f = self.ConvBlock1(x_f)  # [B,112,3]
        rPPG = self.ConvBlock2(x_f) #[B,1,3]
        rPPG = rPPG.squeeze(1)  #[B3]
        return rPPG

###�������������ģ��ļ���

# My Convolution Block
class Conv2(nn.Module):
    def __init__(self, inplane, outplane):
        super(Conv2, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.stage0(x)
        return x

class Conv3(nn.Module):
    def __init__(self, inplane, outplane):
        super(Conv3, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.stage0(x)
        return x

class Conv4(nn.Module):
    def __init__(self, inplane, outplane):
        super(Conv4, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.stage0(x)
        return x

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)

class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme  # MAX, MEAN, SUM
        self.conv2 = Conv2(64, 64)
        self.conv3 = Conv3(64, 64)
        self.conv4 = Conv4(64, 16)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)
        self.ffn = FFN(16*28*28, 4096, 1024)

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def view_tensor(self, tensor_var):
        tensor_1d = tensor_var.clone().view(tensor_var.size(0), -1)
        return tensor_1d


    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)
        sh_feature = self.operate(self.conv3, outs)
        sp_feature_rgb = self.conv3(outs[0])
        sp_feature_msr = self.conv3(outs[1])
        # ���о������ͨ��ά��
        sh_feature = self.operate(self.conv4, sh_feature)
        sp_feature_rgb = self.conv4(sp_feature_rgb)
        sp_feature_msr = self.conv4(sp_feature_msr)
        ##�������ӻ�
        # feature_map = sp_feature_msr[0, 63, :, :].detach().numpy()
        # plt.imshow(feature_map)
        # plt.show()

        # Feature reconstruction
        sh_feature_1d = self.operate(self.ffn,self.operate(self.view_tensor,sh_feature))
        sp_feature_rgb_1d = self.ffn(sp_feature_rgb.clone().view(sp_feature_rgb.size(0), -1))
        sp_feature_msr_1d = self.ffn(sp_feature_msr.clone().view(sp_feature_msr.size(0), -1))

        # Feature fusion
        if self.fuse_scheme == 0:  # MAX
            sh_feature_final = self.tensor_max(sh_feature_1d)
        elif self.fuse_scheme == 1:  # SUM
            sh_feature_final = self.tensor_sum(sh_feature_1d)
        elif self.fuse_scheme == 2:  # MEAN
            sh_feature_final = self.tensor_mean(sh_feature_1d)
        else:  # Default: MAX
            sh_feature_final = self.tensor_max(sh_feature_1d)

        return sp_feature_rgb_1d,sp_feature_msr_1d,sh_feature_1d,sh_feature_final

class MyIFCNN(nn.Module):
    def __init__(self, fuse_scheme=2):
        super(MyIFCNN, self).__init__()
        # pretrained resnet101
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # our model
        self.model = IFCNN(self.resnet, fuse_scheme=fuse_scheme)

    def forward(self, x, y):
        return self.model(x, y)


class RMcom_former(nn.Module):
    def __init__(self):
        super(RMcom_former, self).__init__()
        self.model1 = MyIFCNN()
        self.model2 = TimeSformer_update()
        self.ffn = FFN(3072, 1024, 768)  # �����������ά��Ϊ3072����ά���Ϊ768
    def forward(self, x, y):
        B, T, C, H, W = x.shape
        # Reshape input to [B*T, C, H, W]
        x_reshaped = x.clone().view(-1, 3, 224, 224)  # B,T���������, B*T,C,H,W
        y_reshaped = y.clone().view(-1, 3, 224, 224)
        sp_feature_rgb_1d, sp_feature_msr_1d, sh_feature_1d, sh_feature_final = self.model1(x_reshaped, y_reshaped)
        fusion_features = torch.cat([sh_feature_final, sp_feature_rgb_1d, sp_feature_msr_1d], dim=1)
        fusion_features = self.ffn(fusion_features)
        fusion_features_reshaped = fusion_features.clone().view(B, T, 768)
        rPPG = self.model2(fusion_features_reshaped)
        return sp_feature_rgb_1d, sp_feature_msr_1d, sh_feature_1d, sh_feature_final, rPPG