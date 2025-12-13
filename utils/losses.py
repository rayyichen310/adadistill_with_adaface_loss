import math

import torch
from torch import nn
import torch.distributed as dist


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class MLLoss(nn.Module):
    def __init__(self, s=64.0):
        super(MLLoss, self).__init__()
        self.s = s
        self.kernel = None

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta.mul_(self.s)
        return cos_theta


class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class ACosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(ACosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, embeddings_t, label, momentum):
        embeddings = l2_norm(embeddings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            self.kernel[:, label] = momentum * self.kernel[:, label] + (1.0 - momentum) * (embeddings_t.T)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class AdaptiveACosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35, adaptive_weighted_alpha=True):
        super(AdaptiveACosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.cosine_sim = torch.nn.CosineSimilarity()
        self.adaptive_weighted_alpha = adaptive_weighted_alpha

    def forward(self, embeddings, embeddings_t, label):
        embeddings = l2_norm(embeddings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            cos_theta_tmp = self.cosine_sim(embeddings, embeddings_t)
            cos_theta_tmp = cos_theta_tmp.clamp(-1, 1)
            if self.adaptive_weighted_alpha:
                lam = self.cosine_sim(self.kernel[:, label].T, embeddings_t).clamp(1e-6, 1).view(-1, 1)
                target_logit = cos_theta_tmp.view(-1, 1) * lam
            else:
                target_logit = cos_theta_tmp.view(-1, 1)
            target_logit_mean = target_logit.clamp(1e-6, 1.0).T
            self.kernel[:, label] = target_logit_mean * self.kernel[:, label] + (1.0 - target_logit_mean) * (
                embeddings_t.T
            )
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret, target_logit_mean.mean(), lam.mean(), cos_theta_tmp.mean()


class AdaptiveAArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35, adaptive_weighted_alpha=True):
        super(AdaptiveAArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.cosine_sim = torch.nn.CosineSimilarity()
        self.adaptive_weighted_alpha = adaptive_weighted_alpha

    def forward(self, embeddings, embeddings_t, label):
        embeddings = l2_norm(embeddings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            cos_theta_tmp = self.cosine_sim(embeddings, embeddings_t)
            cos_theta_tmp = cos_theta_tmp.clamp(-1, 1)
            if self.adaptive_weighted_alpha:
                lam = self.cosine_sim(self.kernel[:, label].T, embeddings_t).clamp(1e-6, 1).view(-1, 1)
                target_logit = cos_theta_tmp.view(-1, 1) * lam
            else:
                target_logit = cos_theta_tmp.view(-1, 1)
            target_logit_mean = target_logit.clamp(1e-6, 1.0).T
            self.kernel[:, label] = target_logit_mean * self.kernel[:, label] + (1.0 - target_logit_mean) * (
                embeddings_t.T
            )
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, target_logit_mean.mean(), lam.mean(), cos_theta_tmp.mean()


class AdaptiveAAdaFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        s=64.0,
        m=0.4,
        h=0.333,
        t_alpha=0.01,
        adaptive_weighted_alpha=True,
        use_geom_margin=False,
        geom_margin_w=0.2,
        geom_margin_k=1.0,
        geom_margin_mask=0.8,
        geom_margin_baseline=0.25,
        geom_margin_warmup_epoch=0,
    ):
        super(AdaptiveAAdaFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.cosine_sim = torch.nn.CosineSimilarity()
        self.adaptive_weighted_alpha = adaptive_weighted_alpha
        self.eps = 1e-3
        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)
        self.use_geom_margin = use_geom_margin
        self.geom_margin_w = geom_margin_w
        self.geom_margin_k = geom_margin_k
        self.geom_margin_mask = geom_margin_mask
        self.geom_margin_baseline = geom_margin_baseline
        self.geom_margin_warmup_epoch = geom_margin_warmup_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, embeddings, embeddings_t, label, norms):
        embeddings = l2_norm(embeddings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            cos_theta_tmp = self.cosine_sim(embeddings, embeddings_t)
            cos_theta_tmp = cos_theta_tmp.clamp(-1, 1)
            lam = self.cosine_sim(self.kernel[:, label].T, embeddings_t).clamp(1e-6, 1).view(-1, 1)
            if self.adaptive_weighted_alpha:
                alpha = cos_theta_tmp.view(-1, 1) * lam
            else:
                alpha = cos_theta_tmp.view(-1, 1)
            alpha_clamped = alpha.clamp(1e-6, 1.0)
            self.kernel[:, label] = alpha_clamped.T * self.kernel[:, label] + (1.0 - alpha_clamped.T) * (
                embeddings_t.T
            )
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
        safe_norms = norms.detach().clamp(0.001, 100)
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        # 品質項（AdaFace 的 norm-based scaler）
        margin_scaler_q = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler_q = margin_scaler_q * self.h

        # 可選的 geometry-aware 加成：只對落後樣本增加 margin
        if self.use_geom_margin:
            cos_st = cos_theta_tmp.view(-1, 1)
            
            # 方案 C (Fully Adaptive) - Simplified Formula:
            # Penalty = ReLU(Conf_t - cos_st) * Conf_t
            # 只要 Student 跟 Teacher 的相似度 (cos_st) 低於 Teacher 的自信度 (lam)，就開始懲罰。
            # 懲罰力道隨 Teacher 自信度 (lam) 增加。
            
            geom_delta = (lam - cos_st).clamp(min=0.0)
            geom_delta = geom_delta * lam * self.geom_margin_k
            
            geom_delta = geom_delta.view(-1)  # 展平为一维向量

            # 幾何加成的 warmup：前 geom_margin_warmup_epoch 逐步放大權重
            geom_w_eff = self.geom_margin_w
            if self.geom_margin_warmup_epoch > 0:
                scale = self.current_epoch / self.geom_margin_warmup_epoch
                scale = min(1.0, max(0.0, scale))
                geom_w_eff = self.geom_margin_w * scale

            margin_scaler = margin_scaler_q + geom_w_eff * geom_delta
            
            # 記錄幾何懲罰的統計值
            self.last_geom_penalty = geom_delta.detach().mean()
            self.last_geom_weighted = (geom_w_eff * geom_delta).detach().mean()
        else:
            margin_scaler = margin_scaler_q
            self.last_geom_penalty = torch.tensor(0.0, device=embeddings.device)
            self.last_geom_weighted = torch.tensor(0.0, device=embeddings.device)

        margin_scaler = torch.clamp(margin_scaler, -1, 1)
        margin_scaler = margin_scaler.view(-1)  # 确保是一维向量
        batch_size = label.size(0)
        index = torch.arange(batch_size, device=label.device)
        target_logit = cosine[index, label]
        theta = target_logit.acos()
        g_angular = self.m * margin_scaler * -1
        # 确保 g_angular 是一维向量且长度匹配 theta
        if g_angular.dim() > 1:
            g_angular = g_angular.view(-1)
        theta_m = torch.clamp(theta + g_angular, min=self.eps, max=math.pi - self.eps)
        margin_final_logit = theta_m.cos()
        g_add = self.m + (self.m * margin_scaler)
        margin_final_logit = margin_final_logit - g_add
        cosine[index, label] = margin_final_logit
        logits = cosine * self.s
        return logits, margin_scaler.mean(), lam.mean(), cos_theta_tmp.mean(), self.last_geom_penalty, self.last_geom_weighted


@torch.no_grad()
def all_gather_tensor(input_tensor, dim=0):
    world_size = dist.get_world_size()
    tensor_size = torch.tensor([input_tensor.shape[0]], dtype=torch.int64).cuda()
    tensor_size_list = [torch.ones_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(tensor_list=tensor_size_list, tensor=tensor_size, async_op=False)
    max_size = torch.cat(tensor_size_list, dim=0).max()
    padded = torch.empty(max_size.item(), *input_tensor.shape[1:], dtype=input_tensor.dtype).cuda()
    padded[:input_tensor.shape[0]] = input_tensor
    padded_list = [torch.ones_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list=padded_list, tensor=padded, async_op=False)
    slices = []
    for ts, t in zip(tensor_size_list, padded_list):
        slices.append(t[:ts.item()])
    return torch.cat(slices, dim=0)


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


