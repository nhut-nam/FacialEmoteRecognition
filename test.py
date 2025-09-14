import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.davit import ChannelBlock


class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out

class SMAL(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(SMAL, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout)

        # self-attention branch (simple conv projection)
        self.self_attn = nn.Conv2d(channels, channels, kernel_size=1)

        # mutual attention branch (1x1 conv)
        self.mutual_attn = nn.Conv2d(channels, channels, kernel_size=1)

        # SAM + CBAM
        self.sam = SpatialAttention(3)
        self.cbam = CBAM(channels, 16, 3)

    def forward(self, f1, f2):
        """
        f1: feature từ Net1
        f2: feature từ Net2
        return: f1_hat, f2_hat (sau khi self-mutual attention learning)
        """
        # --- Net1 branch ---
        f1_self = self.self_attn(f1)
        f2_self = self.self_attn(f2)
        f2_cbam = self.cbam(f2)
        f1_mutual = self.mutual_attn(self.dropout(f1)) * f2_cbam  # nhận thông tin từ f2
        f1_sam = self.sam(f1)
        f1_hat = f1 + f1_self * f1_sam + f1_mutual

        # --- Net2 branch ---
        f2_mutual = self.mutual_attn(self.dropout(f2)) * f1_sam # nhận thông tin từ f1
        f2_hat = f2 + f2_self * f2_cbam + f2_mutual

        return f1_hat, f2_hat

class FusionConcat(nn.Module):
    def __init__(self, c1=96, c2=64, out_c=64):
        super().__init__()
        self.conv = nn.Conv2d(c1 + c2, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.attn = ChannelBlock(out_c, num_heads=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        # giả sử f1 và f2 có cùng H, W
        x = torch.cat([f1, f2], dim=1)  # (B, 96+64, H, W)
        return self.act(self.attn(self.bn(self.conv(x))))

class CrossAttention(nn.Module):
    def __init__(self, in_dim_q, in_dim_kv, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(in_dim_q, embed_dim)
        self.k_proj = nn.Linear(in_dim_kv, embed_dim)
        self.v_proj = nn.Linear(in_dim_kv, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, q, kv):
        B, Cq, H, W = q.shape
        _, Ckv, Hk, Wk = kv.shape
        Nq, Nk = H * W, Hk * Wk

        # (B, C, H, W) -> (B, N, C)
        q = q.flatten(2).transpose(1, 2)    # (B, Nq, Cq)
        kv = kv.flatten(2).transpose(1, 2)  # (B, Nk, Ckv)

        Q = self.q_proj(q)   # (B, Nq, embed_dim)
        K = self.k_proj(kv)  # (B, Nk, embed_dim)
        V = self.v_proj(kv)  # (B, Nk, embed_dim)

        attn = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn @ V  # (B, Nq, embed_dim)

        # Đưa về lại (B, embed_dim, H, W)
        out = out.transpose(1, 2).reshape(B, -1, H, W)
        return out

# -------------------
# Network Wrapper
# -------------------
class DualNet(nn.Module):
    def __init__(self, backbone1, backbone2, num_classes):
        super(DualNet, self).__init__()
        self.net1 = backbone1
        self.net2 = backbone2

        self.stn = self.net1.stn

        # self.down_sample1 = nn.Sequential(*[nn.Conv2d(64, 96, kernel_size=1),
        #                                     LayerNorm(96, eps=1e-6, data_format="channels_first")])
        # self.down_sample2 = nn.Sequential(*[nn.Conv2d(128, 192, kernel_size=1),
        #                                     LayerNorm(192, eps=1e-6, data_format="channels_first")])
        # self.down_sample3 = nn.Sequential(*[nn.Conv2d(256, 384, kernel_size=1),
        #                                     LayerNorm(384, eps=1e-6, data_format="channels_first")])
        # self.down_sample4 = nn.Sequential(*[nn.Conv2d(512, 786, kernel_size=1),
        #                                     LayerNorm(786, eps=1e-6, data_format="channels_first")])

        # Thêm SMAL sau backbone1
        # self.smal1 = SMAL(channels=96)  # sửa số channel cuối backbone1 cho đúng
        self.smal2 = SMAL(channels=192)
        # self.smal3 = SMAL(channels=384)
        self.smal4 = SMAL(channels=768)

        # self.smal1 = FusionConcat(96, 96, 96)
        # self.smal2 = FusionConcat(192, 192, 192)
        # self.smal3 = FusionConcat(384, 384, 384)
        # self.smal4 = CrossAttention(768, 768, 768)


        self.fc1 = nn.Linear(768, num_classes)
        self.fc2 = nn.Linear(768, num_classes)

    def forward(self, x, labels=None):
        f1 = self.net1.forward_feature(x, 0)
        f2 = self.net2.forward_feature(x, 0)
        # f2 = self.smal1(f1, f2)

        f1 = self.net1.forward_feature(f1, 1)
        f2 = self.net2.forward_feature(f2, 1)
        # f1, f2 = self.smal2(f1, f2)

        f1 = self.net1.forward_feature(f1, 2)
        f2 = self.net2.forward_feature(f2, 2)
        # f2 = self.smal3(f1, f2)

        f1 = self.net1.forward_feature(f1, 3)
        f2 = self.net2.forward_feature(f2, 3)
        # f1, f2 = self.smal4(f1, f2)

        # logits1
        f1 = self.net1.norm(f1.mean([-2, -1]))
        logits1 = self.fc1(f1)

        # logits2
        f2 = self.net2.norm(f2.mean([-2, -1]))
        logits2 = self.fc2(f2)

        # ensemble trung bình
        logits = (logits1 + logits2) / 2
        preds = torch.argmax(logits, dim=1)

        if labels is not None:
            loss = diffdml_loss(
                logits1, logits2, f1, f2, labels, lambda_kl=1.0
            )
            return preds, logits, loss

        return preds, logits


# -------------------
# Loss Function
# -------------------
def bidirectional_kl_loss(p_logits, q_logits, reduction="batchmean", eps=1e-8, T=2.0):
    # log-prob cho KL-div
    p_log = (F.softmax(p_logits / T, dim=1) + eps).clamp(min=eps)
    q_log = (F.softmax(q_logits / T, dim=1) + eps).clamp(min=eps)

    # prob cho KL-div
    p_prob = p_log.exp()
    q_prob = q_log.exp()

    # KL(P||Q) + KL(Q||P)
    kl_pq = F.kl_div(p_log, q_prob, reduction=reduction)
    kl_qp = F.kl_div(q_log, p_prob, reduction=reduction)

    return kl_pq + kl_qp

def feature_cosine_diversity(fm, fw, eps=1e-8):
    """
    fm, fw: tensors (B, C, H, W) or (B, D) embeddings
    returns mean cosine similarity across batch (we will minimize it).
    """
    B = fm.shape[0]
    # flatten per sample
    fm_flat = fm.view(B, -1)
    fw_flat = fw.view(B, -1)
    # zero-center (optional, often stabilizes)
    fm_flat = fm_flat - fm_flat.mean(dim=1, keepdim=True)
    fw_flat = fw_flat - fw_flat.mean(dim=1, keepdim=True)
    # norms
    fm_norm = fm_flat.norm(dim=1).clamp(min=eps)
    fw_norm = fw_flat.norm(dim=1).clamp(min=eps)
    cos = (fm_flat * fw_flat).sum(dim=1) / (fm_norm * fw_norm)
    return cos.mean()  # minimize this to reduce similarity

def diffdml_loss(logits_m, logits_w, feats_m, feats_w, targets,
                 lambda_kl=1.0, mu_div=0.1, label_smoothing=0.0):
    # supervised CE (with optional label smoothing)
    ce_m = F.cross_entropy(logits_m, targets, label_smoothing=label_smoothing)
    ce_w = F.cross_entropy(logits_w, targets, label_smoothing=label_smoothing)
    # symmetric KL (stable impl)
    kl = bidirectional_kl_loss(logits_m, logits_w)

    # diversity on features (cosine). minimize cosine -> features become less similar
    div = feature_cosine_diversity(feats_m, feats_w)

    loss = ce_m + ce_w + lambda_kl * kl + mu_div * div
    return loss

def get_model(backbone_a, backbone_b, num_classes):
    return DualNet(backbone_a, backbone_b, num_classes)