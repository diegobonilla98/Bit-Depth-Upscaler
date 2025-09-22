import torch
import torch.nn as nn
from typing import List
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )
    def forward(self, x): return self.b(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.d = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.d(x)

class FirstUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.u = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x): return self.u(x)

class FirstDequantUNet(nn.Module):
    """
    Learns to 'dequantize' 8b/channel inputs toward 16b/channel targets.
    Residual prediction with zero-inited head (identity start).
    Output is in [0,1] sRGB, same dynamic range as target.
    """
    def __init__(self, channels=3, width=32):
        super().__init__()
        c = width
        self.enc1 = ConvBlock(channels, c)
        self.down1 = Down(c)
        self.enc2 = ConvBlock(c, c*2)
        self.down2 = Down(c*2)
        self.enc3 = ConvBlock(c*2, c*4)

        self.up2  = FirstUp(c*4, c*2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1  = FirstUp(c*2, c)
        self.dec1 = ConvBlock(c*2, c)

        self.head = nn.Conv2d(c, channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # The true underlying 16b value lies within ±1 LSB (8-bit) of the 8-bit level,
        # so allowing offsets up to ~1/255 is theoretically sufficient.
        self.offset_scale = 1.0 / 255.0

    def forward(self, x_srgb01):
        e1 = self.enc1(x_srgb01)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        offset = torch.tanh(self.head(d1)) * self.offset_scale
        y = torch.clamp(x_srgb01 + offset, 0.0, 1.0)
        return y, offset

class SecondUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.u = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.u(x)

OFFSET_CAP_LSB = 2.0
STRICT_DEQUANT_BIN = False

class SecondDequantUNet(nn.Module):
    """
    Learns to 'dequantize' 8b/channel inputs toward 16b/channel targets.
    Residual prediction with zero-inited head (identity start).
    Output is in [0,1] sRGB, same dynamic range as target.
    """
    def __init__(self, channels=3, width=32):
        super().__init__()
        c = width
        self.enc1 = ConvBlock(channels, c)
        self.down1 = Down(c)
        self.enc2 = ConvBlock(c, c*2)
        self.down2 = Down(c*2)
        self.enc3 = ConvBlock(c*2, c*4)

        self.up2  = SecondUp(c*4, c*2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1  = SecondUp(c*2, c)
        self.dec1 = ConvBlock(c*2, c)

        self.head = nn.Conv2d(c, channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # The true underlying 16b value lies within ±1 LSB (8-bit) of the 8-bit level,
        # so allowing offsets up to ~1/255 is theoretically sufficient.
        self.offset_scale = float(OFFSET_CAP_LSB) / 255.0

    def forward(self, x_srgb01):
        e1 = self.enc1(x_srgb01)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        offset = torch.tanh(self.head(d1)) * self.offset_scale
        y = torch.clamp(x_srgb01 + offset, 0.0, 1.0)
        if STRICT_DEQUANT_BIN:
            q = torch.round(x_srgb01 * 255.0) / 255.0
            y = torch.clamp(y, q - (0.5/255.0), q + (0.5/255.0))
            y = torch.clamp(y, 0.0, 1.0)
        return y, offset


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x / keep * mask

class SEBlock(nn.Module):
    """Squeeze-and-Excitation on channels."""
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, c // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.mlp(self.avg(x))
        return x * w

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block:
      dwconv(7x7) -> (optional SE) -> LayerNorm(channels-last) -> Linear up -> GELU -> Linear down -> gamma -> residual + DropPath
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, ls_init_value: float = 1e-6,
                 drop_path: float = 0.0, use_se: bool = True, se_reduction: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.use_se = use_se
        self.se = SEBlock(dim, reduction=se_reduction) if use_se else nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(mlp_ratio * dim)
        self.pw1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.se(x)
        # channels-last for LayerNorm + MLP
        x = x.permute(0, 2, 3, 1)  # NHWC
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NCHW
        x = residual + self.drop_path(x)
        return x

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """B,C,H,W -> B*nW, C, ws, ws"""
    B, C, H, W = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H += pad_h
        W += pad_w
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, ws, ws)
    return x

def _window_unpartition(xw: torch.Tensor, ws: int, H: int, W: int, B: int) -> torch.Tensor:
    """B*nW,C,ws,ws -> B,C,H,W (crop pad)"""
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    Hp = H + pad_h
    Wp = W + pad_w
    xw = xw.view(B, Hp // ws, Wp // ws, -1, ws, ws)
    xw = xw.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, Hp, Wp)
    return xw[:, :, :H, :W]

class WindowAttention(nn.Module):
    """
    Windowed Multi-Head Self-Attention over ws×ws tokens (simplified, no shifts).
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = int(window_size)
        if num_heads > 0:
            assert dim % num_heads == 0, f"num_heads ({num_heads}) must divide dim ({dim})"
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W
        B, C, H, W = x.shape
        ws = self.ws
        xw = _window_partition(x, ws)  # B*nW, C, ws, ws
        Bn = xw.shape[0]
        xw = xw.flatten(2).transpose(1, 2)  # (B*nW, ws*ws, C)
        xw = self.norm(xw)
        out, _ = self.mha(xw, xw, xw, need_weights=False)
        out = self.attn_drop(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(Bn, C, ws, ws)
        y = _window_unpartition(out, ws, H, W, B)  # B,C,H,W
        return y

class AttnBlock(nn.Module):
    """Pre-norm attention + MLP (ConvNeXt MLP) with residuals and DropPath."""
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.attn = WindowAttention(dim, num_heads=num_heads, window_size=window_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ffn = ConvNeXtBlock(dim, mlp_ratio=mlp_ratio, ls_init_value=1e-6, drop_path=drop_path, use_se=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(x))
        x = self.ffn(x)  # already residual inside
        return x

class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, block_dps: List[float], attn_dp: float,
                 use_se: bool, attn_heads: int = 0, attn_ws: int = 8):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2) if in_ch != out_ch else nn.Identity()
        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXtBlock(out_ch, mlp_ratio=4.0, drop_path=block_dps[i], use_se=use_se))
        self.blocks = nn.Sequential(*blocks)
        self.use_attn = attn_heads > 0
        self.attn = AttnBlock(out_ch, attn_heads, window_size=attn_ws, drop_path=attn_dp) if self.use_attn else nn.Identity()

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        x = self.attn(x)
        return x

class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, depth: int, block_dps: List[float], attn_dp: float,
                 use_se: bool, attn_heads: int = 0, attn_ws: int = 8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXtBlock(out_ch, mlp_ratio=4.0, drop_path=block_dps[i], use_se=use_se))
        self.blocks = nn.Sequential(*blocks)
        self.use_attn = attn_heads > 0
        self.attn = AttnBlock(out_ch, attn_heads, window_size=attn_ws, drop_path=attn_dp) if self.use_attn else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        # match spatial in case of odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        x = self.attn(x)
        return x

class ThirdDequantUNet(nn.Module):
    """
    Powerful U-Net for 8->16 dequantization.
    - ConvNeXt-style residual blocks + SE
    - Windowed MSA in high-level stages (memory friendly)
    - Stochastic depth (drop-path)
    - Identity-start, bounded residual head with tanh scaling
    """
    def __init__(
        self,
        channels: int = 3,
        base_width: int = 64,
        depths: List[int] = (3, 4, 6),
        attn_heads: List[int] = (0, 4, 8),
        attn_window: int = 8,
        use_se: bool = True,
        drop_path_rate: float = 0.1,
        offset_cap_lsb: float = 2.0,
        strict_dequant_bin: bool = False,
    ):
        super().__init__()
        assert len(depths) == 3 and len(attn_heads) == 3, "Expect 3-stage encoder/decoder config"
        c1, c2, c3 = base_width, base_width * 2, base_width * 4

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # drop-path schedule (count every residual/attn block)
        n_enc_blocks = sum(depths)
        n_dec_blocks = depths[1] + depths[0]
        n_attn = int(attn_heads[0] > 0) + int(attn_heads[1] > 0) + int(attn_heads[2] > 0)  # encoder attn
        n_attn += int(attn_heads[1] > 0) + int(attn_heads[0] > 0)  # decoder attn
        total_blocks = n_enc_blocks + n_dec_blocks + n_attn + 3  # +3 for (bn conv, bn attn, bn conv)
        dpr = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()
        it = iter(dpr)

        # encoder
        enc1_block_dps = [next(it) for _ in range(depths[0])]
        enc1_attn_dp = (next(it) if attn_heads[0] > 0 else 0.0)
        self.enc1 = EncoderStage(c1, c1, depths[0], enc1_block_dps, enc1_attn_dp, use_se, attn_heads[0], attn_window)

        enc2_block_dps = [next(it) for _ in range(depths[1])]
        enc2_attn_dp = (next(it) if attn_heads[1] > 0 else 0.0)
        self.enc2 = EncoderStage(c1, c2, depths[1], enc2_block_dps, enc2_attn_dp, use_se, attn_heads[1], attn_window)

        enc3_block_dps = [next(it) for _ in range(depths[2])]
        enc3_attn_dp = (next(it) if attn_heads[2] > 0 else 0.0)
        self.enc3 = EncoderStage(c2, c3, depths[2], enc3_block_dps, enc3_attn_dp, use_se, attn_heads[2], attn_window)

        # bottleneck (validated heads)
        bn_heads = attn_heads[-1] if attn_heads[-1] > 0 else 8
        assert c3 % bn_heads == 0, f"bottleneck heads {bn_heads} must divide {c3}"
        self.bottleneck = nn.Sequential(
            ConvNeXtBlock(c3, drop_path=next(it), use_se=use_se),
            AttnBlock(c3, num_heads=bn_heads, window_size=attn_window, drop_path=next(it)),
            ConvNeXtBlock(c3, drop_path=next(it), use_se=use_se),
        )

        # decoder
        dec2_block_dps = [next(it) for _ in range(depths[1])]
        dec2_attn_dp = (next(it) if attn_heads[1] > 0 else 0.0)
        self.dec2 = DecoderStage(c3, c2, c2, depths[1], dec2_block_dps, dec2_attn_dp, use_se, attn_heads[1], attn_window)

        dec1_block_dps = [next(it) for _ in range(depths[0])]
        dec1_attn_dp = (next(it) if attn_heads[0] > 0 else 0.0)
        self.dec1 = DecoderStage(c2, c1, c1, depths[0], dec1_block_dps, dec1_attn_dp, use_se, attn_heads[0], attn_window)

        # head
        self.head = nn.Conv2d(c1, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.offset_scale = float(offset_cap_lsb) / 255.0
        self.strict_bin = bool(strict_dequant_bin)

    def forward(self, x_srgb01: torch.Tensor):
        # encoder
        x0 = self.stem(x_srgb01)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # bottleneck
        b = self.bottleneck(e3)

        # decoder
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)

        # offset prediction
        offset = torch.tanh(self.head(d1)) * self.offset_scale
        y = torch.clamp(x_srgb01 + offset, 0.0, 1.0)

        if self.strict_bin:
            q = torch.round(x_srgb01 * 255.0) / 255.0
            y = torch.clamp(y, q - 0.5/255.0, q + 0.5/255.0)
            y = torch.clamp(y, 0.0, 1.0)

        return y, offset

