"""
================================================================================
Phy-Bridge-YOLO26 创新模块
================================================================================

此文件包含全部创新模块，直接放入 ultralytics 的 modules/ 目录即可。
无需修改本文件内容。
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 优先使用 ultralytics 已有的 Conv，不存在时 fallback
try:
    from ultralytics.nn.modules import Conv
except ImportError:
    class Conv(nn.Module):
        default_act = nn.SiLU()
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            p = k // 2 if p is None else p
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        def forward(self, x): return self.act(self.bn(self.conv(x)))


# ==============================================================================
# 1. 物理预处理 PIIP
# ==============================================================================
class PIIP(nn.Module):
    """Physics-Informed Infrared Preprocessor"""
    def __init__(self, c):
        super().__init__()
        self.est = nn.Sequential(Conv(c, c // 2, 3), Conv(c // 2, c // 4, 3), nn.Conv2d(c // 4, 2, 3, 1, 1), nn.Sigmoid())
        self.rest = nn.Sequential(Conv(c * 2, c, 3), Conv(c, c, 3))
    def forward(self, x):
        p = self.est(x)
        t, k = p[:, 0:1], p[:, 1:2]
        xr = x * (1 + t) - k * x
        return self.rest(torch.cat([x, xr], 1)), p


# ==============================================================================
# 2. 边缘提取 & 跨模态边缘增强 CMEE
# ==============================================================================
class EdgeEx(nn.Module):
    """多尺度边缘提取器"""
    def __init__(self, c):
        super().__init__()
        self.k3 = nn.Conv2d(c, c // 2, 3, 1, 1, bias=False)
        self.k5 = nn.Conv2d(c, c // 2, 5, 1, 2, bias=False)
        self.k7 = nn.Conv2d(c, c // 2, 7, 1, 3, bias=False)
        self.fuse = Conv(c * 3 // 2, c, 1)
        self.attn = nn.Sequential(Conv(c, c, 3), nn.Conv2d(c, c, 3, 1, 1), nn.Sigmoid())
    def forward(self, x):
        e = self.fuse(torch.cat([self.k3(x), self.k5(x), self.k7(x)], 1))
        return x * (1 + self.attn(e))


class CMEE(nn.Module):
    """Cross-Modal Edge Enhancement"""
    def __init__(self, c):
        super().__init__()
        self.rg, self.ir = EdgeEx(c), EdgeEx(c)
        self.guide = nn.Sequential(Conv(c * 2, c, 3), nn.Conv2d(c, c, 1), nn.Sigmoid())
    def forward(self, r, i):
        r, i = self.rg(r), self.ir(i)
        em = torch.abs(r - F.avg_pool2d(r, 3, 1, 1))
        g = self.guide(torch.cat([em, i], 1))
        return r, i * g + em * (1 - g)


# ==============================================================================
# 3. 可变形对齐 DCMA
# ==============================================================================
class DCMA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.off = nn.Sequential(Conv(c * 2, c, 3), nn.Conv2d(c, 2, 3, 1, 1))
        self.rr, self.ii = Conv(c, c, 3), Conv(c, c, 3)
    def forward(self, r, i):
        o = self.off(torch.cat([r, i], 1))
        # 温度缩放 + 限制在 0.2~0.8，避免极端 0/1
        ar = (torch.sigmoid(o[:, 0:1] / 0.5) * 0.6 + 0.2)
        ai = (torch.sigmoid(o[:, 1:2] / 0.5) * 0.6 + 0.2)
        return self.rr(r * ar + i * (1 - ar)) + r, self.ii(i * ai + r * (1 - ai)) + i


# ==============================================================================
# 4. 双向模态引导 BMGA
# ==============================================================================
class BMGA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ir2rgb = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.SiLU(), nn.Conv2d(c // 4, c, 1))
        self.rgb2ir = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.SiLU(), nn.Conv2d(c // 4, c, 1))
        self.fuse = Conv(c * 2, c, 1)
    def forward(self, r, i):
        # 去掉 Sigmoid，改用 tanh 限制在 -1~1，再缩放到 0.2~0.8
        w1 = (torch.tanh(self.ir2rgb(i)) * 0.3 + 0.5)
        w2 = (torch.tanh(self.rgb2ir(r)) * 0.3 + 0.5)
        return self.fuse(torch.cat([r * w1, i * w2], 1)) + r

# ==============================================================================
# 5. 稀疏跨模态注意力 & 金字塔Transformer PSCMT
# ==============================================================================
class SCMA(nn.Module):
    """Sparse Cross-Modal Attention"""
    def __init__(self, d, nh=8, sr=0.5):
        super().__init__()
        self.nh, self.hd, self.sc = nh, d // nh, d ** -0.5
        self.sr = sr
        self.qp, self.kp, self.vp, self.op = [nn.Linear(d, d) for _ in range(4)]
        self.scorer = nn.Sequential(nn.Linear(d, d // 4), nn.SiLU(), nn.Linear(d // 4, 1))
    def forward(self, rq, ik):
        B, N, C = rq.shape
        k = max(int(N * self.sr), 16)
        rs = torch.topk(self.scorer(rq).squeeze(-1), k, -1)[1]
        is_ = torch.topk(self.scorer(ik).squeeze(-1), k, -1)[1]
        rsx = torch.gather(rq, 1, rs.unsqueeze(-1).expand(-1, -1, C))
        isx = torch.gather(ik, 1, is_.unsqueeze(-1).expand(-1, -1, C))
        Q = self.qp(rsx).view(B, k, self.nh, self.hd).transpose(1, 2)
        K = self.kp(isx).view(B, k, self.nh, self.hd).transpose(1, 2)
        V = self.vp(isx).view(B, k, self.nh, self.hd).transpose(1, 2)
        A = (Q @ K.transpose(-2, -1) * self.sc).softmax(-1)
        out = self.op((A @ V).transpose(1, 2).reshape(B, k, C))
        o = rq.clone()
        o.scatter_(1, rs.unsqueeze(-1).expand(-1, -1, C), out)
        return o


class PSCMT(nn.Module):
    """Pyramid Sparse Cross-Modal Transformer"""
    def __init__(self, d, nh=8):
        super().__init__()
        self.d2, self.d4 = nn.AvgPool2d(2), nn.AvgPool2d(4)
        self.af, self.ah, self.aq = [SCMA(d, nh, 0.5) for _ in range(3)]
        self.fuse = nn.Sequential(Conv(d * 3, d * 2, 1), Conv(d * 2, d, 3))
        self.proj = Conv(d, d, 1)

    def _attn(self, a, r, i, H, W):
        B, C, h, w = r.shape
        rt = r.flatten(2).transpose(1, 2)
        it = i.flatten(2).transpose(1, 2)
        o = a(rt, it).transpose(1, 2).view(B, C, h, w)
        return F.interpolate(o, size=(H, W), mode='bilinear', align_corners=False) if (h, w) != (H, W) else o
    def forward(self, r, i):
        B, C, H, W = r.shape
        return self.proj(self.fuse(torch.cat([self._attn(self.af, r, i, H, W),
                                               self._attn(self.ah, self.d2(r), self.d2(i), H, W),
                                               self._attn(self.aq, self.d4(r), self.d4(i), H, W)], 1)))


# ==============================================================================
# 6. CPCF
# ==============================================================================
class CPCF(nn.Module):
    """Contextual Partial Cross Feature Block"""
    def __init__(self, c):
        super().__init__()
        self.sc = c // 2
        self.loc = nn.Sequential(Conv(self.sc, self.sc, 3), Conv(self.sc, self.sc, 3))
        self.glob = nn.Conv2d(self.sc, self.sc, 1)
        self.bn = nn.BatchNorm2d(self.sc)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.cf = Conv(c, c, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        xl, xg = x[:, :self.sc], x[:, self.sc:]
        lo = self.loc(xl)
        g = self.bn(self.glob(xg)).view(B, self.sc, -1).permute(0, 2, 1)
        a = (g @ g.transpose(1, 2) / math.sqrt(self.sc)).softmax(-1)
        v = xg.view(B, self.sc, -1).permute(0, 2, 1)
        go = xg + self.gamma * (a @ v).permute(0, 2, 1).view(B, self.sc, H, W)
        return self.cf(torch.cat([lo, go], 1)) + x


# ==============================================================================
# 7. 动态多尺度融合 DMFP
# ==============================================================================
class DMFP(nn.Module):
    """Dynamic Multiscale Fusion Pyramid"""
    def __init__(self, c):
        super().__init__()
        self.cr, self.cu, self.cl = Conv(c, c, 3), Conv(c, c, 3), nn.Sequential(nn.AvgPool2d(2), Conv(c, c, 3))
        self.w = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c * 3, 3, 1), nn.Softmax(1))
        self.ref = Conv(c, c, 3)
    def forward(self, x, u=None, l=None):
        f = [self.cr(x)]
        f.append(self.cu(
            F.interpolate(u, size=x.shape[-2:], mode='bilinear', align_corners=False)) if u is not None else self.cr(x))
        f.append(F.interpolate(self.cl(l), size=x.shape[-2:], mode='bilinear',
                               align_corners=False) if l is not None else self.cr(x))
        w = self.w(torch.cat(f, 1))
        return self.ref(sum(ww * ff for ww, ff in zip(w.chunk(3, 1), f)))


# ==============================================================================
# 8. 浅层检测层 SFDL
# ==============================================================================
class SFDL(nn.Module):
    """Shallow Feature Detection Layer"""
    def __init__(self, c, nc, reg_max=1):
        super().__init__()
        self.rf = nn.Sequential(Conv(c, c // 2, 3), Conv(c // 2, c // 2, 3))
        self.st = Conv(c // 2, c // 2, 1)
        self.c2 = nn.Sequential(Conv(c // 2, c // 2, 3), nn.Conv2d(c // 2, 4 * reg_max, 1))
        self.c3 = nn.Sequential(Conv(c // 2, c // 2, 3), nn.Conv2d(c // 2, nc, 1))
    def forward(self, x):
        x = self.st(self.rf(x))
        return torch.cat([self.c2(x), self.c3(x)], 1)


# ==============================================================================
# 9. 损失函数
# ==============================================================================
def inner_mpdiou_loss(pred_boxes, target_boxes, eps=1e-7):
    """Inner-MPDIoU"""
    ix1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    iy1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    ix2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    iy2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    pa = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    ta = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    iou = inter / (pa + ta - inter + eps)
    pcx, pcy = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2, (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    tcx, tcy = (target_boxes[:, 0] + target_boxes[:, 2]) / 2, (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    d1 = torch.sqrt((pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2)
    d2 = torch.sqrt((pred_boxes[:, 2] - target_boxes[:, 2]) ** 2 + (pred_boxes[:, 3] - target_boxes[:, 3]) ** 2)
    d3 = torch.sqrt((pred_boxes[:, 2] - target_boxes[:, 2]) ** 2 + (pred_boxes[:, 3] - target_boxes[:, 3]) ** 2)
    d4 = torch.sqrt((pred_boxes[:, 2] - target_boxes[:, 2]) ** 2 + (pred_boxes[:, 3] - target_boxes[:, 3]) ** 2)
    dc = torch.sqrt((pcx - tcx) ** 2 + (pcy - tcy) ** 2)
    mpd = (d1 + d2 + d3 + d4 + dc) / 5.0
    iw = torch.min(pred_boxes[:, 2] - pred_boxes[:, 0], target_boxes[:, 2] - target_boxes[:, 0])
    ih = torch.min(pred_boxes[:, 3] - pred_boxes[:, 1], target_boxes[:, 3] - target_boxes[:, 1])
    inner_area = torch.clamp(iw, min=0) * torch.clamp(ih, min=0)
    inner_ratio = inner_area / (ta + eps)
    return (1 - iou + mpd / 10.0 + (1 - inner_ratio) * 0.5).mean()


def physics_consistency_loss(physics_params):
    """物理一致性损失"""
    t, k = physics_params[:, 0:1], physics_params[:, 1:2]
    tgx = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])
    tgy = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])
    cgx = torch.abs(k[:, :, :, 1:] - k[:, :, :, :-1])
    cgy = torch.abs(k[:, :, 1:, :] - k[:, :, :-1, :])
    return (tgx.mean() + tgy.mean() + cgx.mean() + cgy.mean()) / 4.0 + 0.1 * (torch.clamp(0.1 - t, min=0).mean() + torch.clamp(t - 1.0, min=0).mean())
