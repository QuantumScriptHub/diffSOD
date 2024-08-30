import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net101_v1b


class SalientBranch(nn.Module):
    def __init__(self, backbone, out_channels, **kwargs):
        super(SalientBranch, self).__init__()
        self.backbone_original = backbone
        self.backbone_x = backbone
        self.in_channels = [64, 256, 512, 1024, 2048]
        self.out_channels = out_channels
        self.original_rfb1 = expand_feats(self.in_channels[2], self.out_channels[2])
        self.original_rfb2 = expand_feats(self.in_channels[3], self.out_channels[3])
        self.original_rfb3 = expand_feats(self.in_channels[4], self.out_channels[4])
        self.x_rfb1 = expand_feats(self.in_channels[2], self.out_channels[2])
        self.x_rfb2 = expand_feats(self.in_channels[3], self.out_channels[3])
        self.x_rfb3 = expand_feats(self.in_channels[4], self.out_channels[4])
        self.agg_original = aggregation((out_channels[2], out_channels[3], out_channels[4]))
        self.agg_x = aggregation((out_channels[2], out_channels[3], out_channels[4]))
        self.afim = Block(dim=out_channels[4], num_head=8, mlp_ratio=4)

    def forward_encode(self, original, img_x):
        original1, original2, original3, original4, original5 = self.backbone_original(original)
        img_x1, img_x2, img_x3, img_x4, img_x5 = self.backbone_original(img_x)

        original3 = self.original_rfb1(original3)  # 8
        original4 = self.original_rfb2(original4)  # 16
        original5 = self.original_rfb3(original5)  # 32
        img_x3 = self.x_rfb1(img_x3)  # 8
        img_x4 = self.x_rfb2(img_x4)  # 16
        img_x5 = self.x_rfb3(img_x5)  # 32

        original_agg = self.agg_original(original5, original4, original3)  # 8
        img_x_agg = self.agg_x(img_x5, img_x4, img_x3)                     # 8
        out = self.afim(original_agg, img_x_agg)

        return out


class expand_feats(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(expand_feats, self).__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Sequential(
            nn.Conv2d(channel[2], channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample2 = nn.Sequential(
            nn.Conv2d(channel[2], channel[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample4 = nn.Sequential(
            nn.Conv2d(channel[2], channel[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample5 = nn.Sequential(
            nn.Conv2d(channel[2] + channel[1], channel[2] + channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2] + channel[1]),
            nn.ReLU(inplace=True),
        )

        self.conv_concat2 = nn.Sequential(
            nn.Conv2d(channel[2] + channel[1], channel[2] + channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2] + channel[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_concat3 = nn.Sequential(
            nn.Conv2d(channel[2] + channel[1] + channel[0], channel[2] + channel[1] + channel[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2] + channel[1] + channel[0]),
            nn.ReLU(inplace=True),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(channel[2] + channel[1] + channel[0], channel[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(self.upsample(e4))) \
               * self.conv_upsample3(self.upsample(e3)) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(self.upsample(e3_2))), 1)
        x = self.conv_concat3(e2_2)

        output = self.conv_last(x)

        return output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x

class attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_fore = nn.Linear(dim, dim)
        self.e_back = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim * 2, dim)
        if window != 0:
            self.short_cut_linear = nn.Linear(dim * 2, dim)
            self.kv = nn.Linear(dim, dim)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.proj = nn.Linear(dim * 3, dim)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim, eps=1e-6, data_format="channels_last")

    def forward(self, x, x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        if self.window != 0:
            short_cut = torch.cat([x, x_e], dim=3)  ##########
            short_cut = short_cut.permute(0, 3, 1, 2)  #############

        q = self.q(x)
        cutted_x = self.q_cut(x)
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)

        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = x.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            short_cut = self.pool(short_cut).permute(0, 2, 3, 1)
            short_cut = self.short_cut_linear(short_cut)
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            m = short_cut
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head).permute(0, 1, 4, 2, 3).reshape(
                B, C, self.window, self.window)
            attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        cutted_x = cutted_x * x_e
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn, cutted_x], dim=3)
        else:
            x = torch.cat([x, cutted_x], dim=3)

        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_head, norm_cfg=dict(type='SyncBN', requires_grad=True), mlp_ratio=4.,
                 window=7, ):
        super().__init__()

        layer_scale_init_value = 1e-6
        self.attn = attention(dim, num_head, window=window, norm_cfg=norm_cfg)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_e):
        x = x.permute(0, 2, 3, 1)
        x_e = x_e.permute(0, 2, 3, 1)
        res_x, res_e = x, x_e
        x = self.attn(x, x_e)
        x = res_x + self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x
        x = x + self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x)

        return x

def SalientBranch_Res2Net101(pretrained, **kwargs):
    return SalientBranch(res2net101_v1b(pretrained=pretrained), [64, 128, 256, 512, 1024], **kwargs)
