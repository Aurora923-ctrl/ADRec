import torch.nn as nn
import torch
import torch.nn.functional as F
from common import SiLU, TransformerEncoder
from utils import _extract_into_tensor, exponential_mapping
from step_sample import *
import numpy as np
import math

class DenoisedModel(nn.Module):
    def __init__(self, args):
        super(DenoisedModel, self).__init__()
        self.hidden_size = args.hidden_size
        if args.dif_decoder == 'mlp':
            self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                         SiLU(),
                                         nn.Linear(self.hidden_size * 4, self.hidden_size),
                                         nn.LayerNorm(self.hidden_size),
                                         )
        else:
            self.decoder = TransformerEncoder(args, num_blocks=2, norm_first=False, hidden_size=self.hidden_size, use_rope=args.use_rope)

        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size)
                                        )

        self.lambda_uncertainty = args.lambda_uncertainty

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward_cfg(self, c, x, t, mask_seq, mask_tgt, cfg_scale=1.0):
        cond_eps = self.forward(c, x, t, mask_seq, mask_tgt)
        uncond_eps = self.forward(c, x, t, mask_seq, mask_tgt, condition=False)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt, condition=True):
        if condition is not True:  # CFG
            rep_item = torch.zeros_like(rep_item)
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        lambda_uncertainty = self.lambda_uncertainty  # fixed

        rep_diffu = rep_item + lambda_uncertainty * (x_t + time_emb)

        if isinstance(self.decoder, nn.Sequential):
            rep_diffu = self.decoder(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)

        return rep_diffu

class AdaptiveTimestepScheduler(nn.Module):
    """
    自适应时间步调度器（改进版）
    根据位置重要性确定性映射到时间步，而不是学习映射
    重要性高 -> 采样小t（更精细的去噪）
    重要性低 -> 采样大t（更粗糙的去噪）
    """
    
    def __init__(self, num_timesteps, hidden_size, temperature=1.0):
        super(AdaptiveTimestepScheduler, self).__init__()
        self.num_timesteps = num_timesteps
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        # 可学习的温度参数（可选）
        self.learnable_temperature = nn.Parameter(torch.tensor(temperature))
        self.use_learnable_temp = False
    
    def forward(self, importance):
        """
        根据位置重要性生成时间步采样分布（确定性映射，支持批量/序列维）
        
        Args:
            importance: [B]、[B, 1] 或 [B, L]，数值范围 [0, 1]
            
        Returns:
            distribution: [B, T] 或 [B, L, T] 的时间步采样分布
        """
        if importance.dim() == 1:
            importance = importance.unsqueeze(-1)  # [B, 1]
        
        # 重要性高 -> 采样小t，重要性低 -> 采样大t
        expected_t = (1.0 - importance) * (self.num_timesteps - 1)  # [...]
        expected_t = expected_t.unsqueeze(-1)  # [..., 1]
        
        # 创建以 expected_t 为中心的高斯分布
        t_range = torch.arange(self.num_timesteps, dtype=torch.float32, device=importance.device)
        # 广播到 [..., T]，使其与 expected_t 维度一致（最后一维为 T）
        while t_range.dim() < expected_t.dim():
            t_range = t_range.unsqueeze(0)
        temp = self.learnable_temperature if self.use_learnable_temp else self.temperature
        inv_two_sigma2 = 1.0 / (2.0 * (temp ** 2))
        logits = -((t_range - expected_t) ** 2) * inv_two_sigma2  # [..., T]
        distribution = F.softmax(logits, dim=-1)  # [..., T]
        return distribution
    
    def sample(self, importance):
        """
        根据位置重要性采样时间步（矢量化）
        
        Args:
            importance: [B] 或 [B, L] 的位置重要性分数
            
        Returns:
            t: [B] 或 [B, L] 采样得到的时间步
        """
        probs = self.forward(importance)  # [B, T] 或 [B, L, T]
        if probs.dim() == 2:
            cat = torch.distributions.Categorical(probs=probs)
            return cat.sample()  # [B]
        else:
            B, L, T = probs.shape
            cat = torch.distributions.Categorical(probs=probs.reshape(-1, T))
            t = cat.sample().reshape(B, L)  # [B, L]
            return t

class AdRec(nn.Module):
    def __init__(self, args):
        super(AdRec, self).__init__()

        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        betas = get_named_beta_schedule(args)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])

        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.net = DenoisedModel(args)
        self.independent_diffusion = args.independent
        self.cfg_scale = args.cfg_scale
        self.geodesic = args.geodesic
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False, use_rope=args.use_rope)
        
        # 位置感知的自适应独立扩散相关组件
        self.use_position_aware = getattr(args, 'use_position_aware', False)
        if self.use_position_aware:
            # 位置重要性计算方式
            self.position_importance_mode = getattr(args, 'position_importance_mode', 'hybrid')  # 'learned', 'position_based', 'hybrid'
            
            # 损失感知的位置重要性（关键改进）
            self.use_error_guided_importance = getattr(args, 'use_error_guided_importance', True)
            self.importance_supervision_weight = getattr(args, 'importance_supervision_weight', 0.1)
            
            # 学习的位置重要性网络（可选）
            if self.position_importance_mode in ['learned', 'hybrid']:
                self.position_importance_net = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    SiLU(),
                    nn.Linear(self.hidden_size // 2, 1),
                    nn.Sigmoid()  # 确保重要性在[0, 1]之间
                )
            else:
                self.position_importance_net = None
            
            # 自适应时间步调度器（改进版：使用确定性映射）
            scheduler_temp = getattr(args, 'scheduler_temperature', 2.0)
            self.adaptive_scheduler = AdaptiveTimestepScheduler(
                num_timesteps=self.num_timesteps,
                hidden_size=self.hidden_size,
                temperature=scheduler_temp
            )
            
            # 初始化预测误差缓存（用于误差引导）
            self._last_prediction_error = None

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        
        sqrt_alpha_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

        if self.geodesic:
            x_t = exponential_mapping(x_start, x_t)
        if mask is None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        if self.cfg_scale == 1.:
            x_0 = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        else:
            x_0 = self.net.forward_cfg(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag, self.cfg_scale)
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq, mask_tag)
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        if self.geodesic:
            sample_xt = F.normalize(sample_xt, p=2, dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.ag_encoder(seq, mask_seq)
        noise_x_t = torch.randn_like(tgt)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = torch.tensor([0] * (seq.shape[1] - 1) + [i], device=seq.device).unsqueeze(0).repeat(seq.shape[0], 1)
            noise_x_t = torch.cat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag)
        return noise_x_t

    def independent_diffuse(self, tgt, mask, is_independent=False, seq_context=None, prediction_error=None):
        """
        独立扩散过程
        
        Args:
            tgt: [B, L, H] 目标序列嵌入
            mask: [B, L] 掩码
            is_independent: 是否使用独立扩散
            seq_context: [B, L, H] 序列上下文（可选，用于位置感知）
            prediction_error: [B, L] 预测误差（可选，用于误差引导的位置重要性计算）
            
        Returns:
            x_t: [B, L, H] 加噪后的序列
            t: [B, L] 或 [B] 时间步
            position_importance: [B, L] 位置重要性（如果使用位置感知）
        """
        if is_independent:
            # 位置感知的自适应独立扩散
            if self.use_position_aware and self.training:
                return self._position_aware_independent_diffuse(tgt, mask, seq_context, prediction_error)
            else:
                # 原始uniform采样
                t, weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
                t = t * mask.reshape(-1).long()
                x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
                return x_t, t
        else:
            # 非独立扩散：所有位置使用相同的时间步
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
            return x_t, t
    
    def _position_aware_independent_diffuse(self, tgt, mask, seq_context=None, prediction_error=None):
        """
        位置感知的自适应独立扩散（改进版 - 损失感知）
        
        改进点：
        1. 使用混合的位置重要性计算（基于位置 + 学习的重要性）
        2. 使用预测误差作为位置重要性的监督信号（最关键）
        3. 使用确定性映射的时间步调度（重要性高 -> 小t）
        4. 支持课程学习式的位置重要性调整
        
        Args:
            tgt: [B, L, H] 目标序列嵌入
            mask: [B, L] 掩码
            seq_context: [B, L, H] 序列上下文（从ag_encoder编码得到）
            prediction_error: [B, L] 预测误差（可选，用于误差引导的重要性计算）
            
        Returns:
            x_t: [B, L, H] 加噪后的序列
            t: [B, L] 每个位置的时间步
            position_importance: [B, L] 位置重要性
        """
        seq_len = tgt.shape[1]
        batch_size = tgt.shape[0]
        device = tgt.device
        
        # 1. 计算位置重要性（改进：损失感知 + 混合方式）
        use_error_guided = getattr(self, 'use_error_guided_importance', True)
        
        if self.position_importance_mode == 'position_based':
            # 方法1：基于位置的简单重要性（最后位置更重要）
            position_weights = torch.linspace(0.3, 1.0, seq_len, device=device)  # [L]
            position_importance = position_weights.unsqueeze(0).repeat(batch_size, 1)  # [B, L]
            # 应用mask并归一化
            position_importance = position_importance * mask
            imp_sum = position_importance.sum(dim=1, keepdim=True) + 1e-8
            position_importance = position_importance / imp_sum
        
        elif self.position_importance_mode == 'learned':
            # 方法2：完全学习的重要性
            if seq_context is None:
                seq_context = tgt
            position_importance = self.position_importance_net(seq_context)  # [B, L, 1]
            position_importance = position_importance.squeeze(-1)  # [B, L]
            # 应用mask并归一化
            position_importance = position_importance * mask
            imp_sum = position_importance.sum(dim=1, keepdim=True) + 1e-8
            position_importance = position_importance / imp_sum
        
        else:  # 'hybrid'
            # 方法3：混合方式（基于位置 + 学习的重要性 + 误差引导）
            # 基于位置的重要性（最后位置更重要）
            position_weights = torch.linspace(0.3, 1.0, seq_len, device=device)  # [L]
            position_based_imp = position_weights.unsqueeze(0).repeat(batch_size, 1)  # [B, L]
            
            # 学习的重要性
            if seq_context is None:
                seq_context = tgt
            if self.position_importance_net is not None:
                learned_imp = self.position_importance_net(seq_context).squeeze(-1)  # [B, L]
                # 混合：0.5 * 位置重要性 + 0.5 * 学习的重要性
                base_importance = 0.5 * position_based_imp + 0.5 * learned_imp
            else:
                base_importance = position_based_imp
            
            # 应用mask到基础重要性
            base_importance = base_importance * mask
            
            # 改进：如果提供了预测误差，使用误差引导
            # 检查prediction_error的维度是否匹配（batch_size和seq_len）
            if use_error_guided and prediction_error is not None:
                # 检查维度是否匹配
                pred_batch, pred_len = prediction_error.shape
                if pred_batch == batch_size and pred_len == seq_len:
                    # 误差大的位置更重要（需要更多训练）
                    # 归一化误差重要性（在有效位置内）
                    error_importance = prediction_error * mask  # 先应用mask
                    error_sum = error_importance.sum(dim=1, keepdim=True) + 1e-8
                    error_importance = error_importance / error_sum  # 归一化
                    
                    # 归一化基础重要性（在有效位置内）
                    base_sum = base_importance.sum(dim=1, keepdim=True) + 1e-8
                    base_importance_norm = base_importance / base_sum
                    
                    # 混合归一化后的重要性：0.6 * 基础重要性 + 0.4 * 误差重要性
                    position_importance = 0.6 * base_importance_norm + 0.4 * error_importance
                else:
                    # 维度不匹配，使用基础重要性
                    base_sum = base_importance.sum(dim=1, keepdim=True) + 1e-8
                    position_importance = base_importance / base_sum
            else:
                # 没有误差引导时，归一化基础重要性
                base_sum = base_importance.sum(dim=1, keepdim=True) + 1e-8
                position_importance = base_importance / base_sum
        
        # 确保无效位置的重要性为0（归一化后已经保证，但再次确认）
        position_importance = position_importance * mask
        
        # 2. 为每个位置自适应采样时间步（矢量化采样）
        # 重要性高 -> 采样小t，重要性低 -> 采样大t
        t = self.adaptive_scheduler.sample(position_importance)  # [B, L]
        # 应用mask
        t = t * mask.long()
        
        # 3. 对每个位置进行加噪
        # 需要将tgt reshape为 [B*L, H]，t reshape为 [B*L]
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])  # [B*L, H]
        t_flat = t.reshape(-1)  # [B*L]
        mask_flat = mask.reshape(-1)  # [B*L]
        
        x_t_flat = self.q_sample(tgt_flat, t_flat, mask=mask_flat)  # [B*L, H]
        x_t = x_t_flat.reshape(*tgt.shape)  # [B, L, H]
        
        return x_t, t, position_importance

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        
        # 位置感知的自适应独立扩散
        use_error_guided = getattr(self, 'use_error_guided_importance', True)
        prediction_error = None
        
        if self.use_position_aware and self.independent_diffusion and self.training:
            if use_error_guided and hasattr(self, '_last_prediction_error') and self._last_prediction_error is not None:
                prediction_error = self._last_prediction_error
            else:
                prediction_error = None
            
            result = self.independent_diffuse(
                item_tag, mask_tag, self.independent_diffusion, 
                seq_context=item_rep, 
                prediction_error=prediction_error
            )
            if isinstance(result, tuple) and len(result) == 3:
                x_t, t, position_importance = result
            else:
                x_t, t = result
                position_importance = None
        else:
            x_t, t = self.independent_diffuse(item_tag, mask_tag, self.independent_diffusion)
            position_importance = None
        
        if self.cfg_scale != 1:
            mask = torch.rand([mask_seq.shape[0],1,1],device=item_rep.device) > 0.7
            item_rep = torch.where(mask,torch.zeros_like(item_rep),item_rep)
        
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        losses = F.mse_loss(denoised_seq,item_tag, reduction='none')* (mask_tag / mask_tag.sum(1,keepdim=True)).unsqueeze(-1)
        losses = losses.sum(1).mean()
        
        # 损失感知的位置重要性监督
        if position_importance is not None and self.training:
            current_prediction_error = F.mse_loss(denoised_seq, item_tag, reduction='none').mean(dim=-1)
            
            if use_error_guided:
                self._last_prediction_error = current_prediction_error.detach()
            
            if use_error_guided:
                error_normalized = current_prediction_error / (current_prediction_error.sum(dim=1, keepdim=True) + 1e-8)
                error_normalized = error_normalized * mask_tag
                
                importance_supervision_weight = getattr(self, 'importance_supervision_weight', 0.05)
                importance_loss = F.mse_loss(
                    position_importance * mask_tag, 
                    error_normalized,
                    reduction='none'
                ).sum(dim=1).mean()
                losses = losses + importance_supervision_weight * importance_loss
            
            importance_entropy = -torch.sum(
                position_importance * torch.log(position_importance + 1e-8) * mask_tag,
                dim=1
            ) / (mask_tag.sum(dim=1) + 1e-8)
            
            entropy_weight = getattr(self, 'importance_entropy_weight', 0.01)
            losses = losses + entropy_weight * (1.0 - importance_entropy.mean())
            
            if self.position_importance_mode == 'hybrid' and mask_tag.sum() > 0:
                B, L = mask_tag.shape
                last_pos_mask = torch.zeros_like(mask_tag)
                has_valid = mask_tag.any(dim=1)
                if has_valid.any():
                    rev = torch.flip(mask_tag, dims=[1])
                    last_from_right = torch.argmax(rev.to(torch.int64), dim=1)
                    last_idx = (L - 1) - last_from_right
                    idx_rows = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
                    idx_cols = last_idx[has_valid].unsqueeze(1)
                    last_pos_mask[idx_rows] = last_pos_mask[idx_rows].scatter(1, idx_cols, 1.0)
                
                last_imp_weight = getattr(self, 'last_importance_weight', 0.005)
                last_imp = (position_importance * last_pos_mask).sum(dim=1).mean()
                losses = losses - last_imp_weight * last_imp
        
        return denoised_seq, losses
