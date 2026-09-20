"""Causal conditional VAE for joint proposals and controlled observation prediction.

A standard-normal latent prior feeds a causal decoder. The recognition network
is used only for training, sees targets under the same causal mask, and shares
public conditioning with the decoder. No simulator or separate dynamics model.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from data.joint_training import encode_batch
from generators.joint_world_model import JointWorldModel, CausalBlock, reduce_query_losses


@dataclass
class CVAEConfig:
    method: str = 'cvae'
    width: int = 256
    layers: int = 6
    heads: int = 8
    posterior_layers: int = 2
    latent_dim: int = 16
    kl_weight: float = 0.01
    kl_warmup_windows: int = 6553600
    free_nats: float = 0.03
    first_action_loss_weight: float = 10.0
    query_loss_weights: tuple = (.45, .25, .20, .10)


class JointCVAE(JointWorldModel):
    """Shared public query API, with one stochastic decoder pass per query."""
    def __init__(self, config, normalization):
        nn.Module.__init__(self)
        self.config = config
        self.normalization = normalization
        if config.method != 'cvae': raise ValueError('Expected cvae method')
        if min(config.width, config.layers, config.heads, config.posterior_layers, config.latent_dim) < 1 or config.width % config.heads:
            raise ValueError('Invalid CVAE dimensions')
        if not math.isfinite(config.kl_weight) or config.kl_weight <= 0 or config.kl_warmup_windows < 0:
            raise ValueError('Positive finite KL weight and nonnegative warmup required')
        if not math.isfinite(config.free_nats) or config.free_nats < 0:
            raise ValueError('Nonnegative finite KL free-nats threshold required')
        if config.first_action_loss_weight < 1 or len(config.query_loss_weights) != 4 or not math.isclose(sum(config.query_loss_weights), 1.):
            raise ValueError('Invalid reconstruction query weights')
        width = config.width
        self.context = nn.Linear(60, width)
        self.constraints = nn.Linear(168, width, bias=False)
        self.quality = nn.Embedding(3, width)
        self.role = nn.Embedding(4, width)
        self.horizon = nn.Embedding(7, width)
        self.position = nn.Parameter(torch.randn(1, 12, width) * .01)
        self.encoder_input = nn.Linear(21, width)
        self.encoder_blocks = nn.ModuleList(CausalBlock(width, config.heads) for _ in range(config.posterior_layers))
        self.encoder_norm = nn.LayerNorm(width)
        self.posterior = nn.Linear(width, 2 * config.latent_dim)
        nn.init.zeros_(self.posterior.weight)
        nn.init.zeros_(self.posterior.bias)
        self.decoder_input = nn.Linear(config.latent_dim + 14, width)
        self.decoder_blocks = nn.ModuleList(CausalBlock(width, config.heads) for _ in range(config.layers))
        self.decoder_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 7)
        self.event = nn.Linear(width, 5)
        self.training_windows = 0
        self.sampling_temperature = 1.0
        self.guidance_scale = 1.0

    @property
    def inference_passes(self):
        return 1

    def inference_work(self,query,H=6):
        if query not in ('proposal','prediction','completion'):raise ValueError('Unknown query role')
        extra=query!='prediction'
        return dict(backbone_evaluations=1+int(extra),imposed_action_reprediction_queries=int(extra),
                    sampling_temperature=self.sampling_temperature,guidance_scale=1.)

    def set_training_progress(self, windows):
        self.training_windows = int(windows)

    def conditioning(self, encoded):
        return (self.context(encoded['context']) + self.completion_context(encoded) + self.quality(encoded['quality']) +
                self.role(encoded['role']) + self.horizon(encoded['horizon']))[:, None] + self.position

    @staticmethod
    def known_values(encoded):
        return torch.where(encoded['known'], encoded['x'], 0.)

    @staticmethod
    def latent_mask(encoded):
        return (encoded['semantic'] & ~encoded['known']).any(-1)

    def encode(self, encoded):
        # Validity and labels enter recognition/loss only, never generation.
        target = torch.where(encoded['loss_mask'], encoded['x'], 0.)
        inputs = torch.cat([target, self.known_values(encoded), encoded['known'].float()], -1)
        hidden = self.encoder_input(inputs) + self.conditioning(encoded)
        for block in self.encoder_blocks: hidden = block(hidden)
        mean, log_variance = self.posterior(self.encoder_norm(hidden)).chunk(2, -1)
        return mean, log_variance.clamp(-8., 4.)

    def forward(self, latent, encoded):
        latent = latent * self.latent_mask(encoded)[..., None]
        inputs = torch.cat([latent, self.known_values(encoded), encoded['known'].float()], -1)
        hidden = self.decoder_input(inputs) + self.conditioning(encoded)
        for block in self.decoder_blocks: hidden = block(hidden)
        hidden = self.decoder_norm(hidden)
        return self.output(hidden), self.event(hidden[:, 1::2])

    def loss(self, batch):
        encoded = encode_batch(batch, self.normalization)
        mean, log_variance = self.encode(encoded)
        latent = mean + torch.exp(.5 * log_variance) * torch.randn_like(mean)
        output, event_logits = self(latent, encoded)
        count = encoded['loss_mask'].sum(-1)
        token_loss = ((output - encoded['x']).square() * encoded['loss_mask']).sum(-1) / count.clamp_min(1)
        reconstruction, metrics = reduce_query_losses(token_loss, encoded, event_logits,
            first_action_weight=self.config.first_action_loss_weight, query_weights=self.config.query_loss_weights)
        coordinate_kl = .5 * (mean.square() + log_variance.exp() - 1. - log_variance)
        valid = encoded['loss_mask'].any(-1)
        token_kl = coordinate_kl.clamp_min(self.config.free_nats).sum(-1)
        row_kl = (token_kl * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        kl = output.new_zeros(())
        for role, weight in enumerate(self.config.query_loss_weights):
            selected = (encoded['role'] == role).to(output.dtype)
            kl = kl + weight * (row_kl * selected).sum() / selected.sum().clamp_min(1)
        warmup = self.config.kl_warmup_windows
        factor = min(1., self.training_windows / warmup) if warmup else 1.
        coefficient = self.config.kl_weight * factor
        total = reconstruction + coefficient * kl
        raw_kl = (coordinate_kl.sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        metrics.update(reconstruction_loss=reconstruction.detach(), kl_loss=kl.detach(),
                       raw_kl_nats_per_token=raw_kl.detach(), kl_coefficient=output.new_tensor(coefficient),
                       posterior_mean_square=(mean.square().sum(-1) * valid).sum().detach() /
                           (valid.sum().clamp_min(1) * self.config.latent_dim))
        return total, metrics

    @torch.no_grad()
    def generate(self, batch, generator=None):
        encoded = encode_batch(batch, self.normalization)
        device = encoded['x'].device
        temperature = float(self.sampling_temperature)
        if not math.isfinite(temperature) or temperature <= 0: raise ValueError('Sampling temperature must be finite and positive')
        latent = temperature * torch.randn(len(encoded['x']), 12, self.config.latent_dim, device=device, generator=generator)
        output, events = self(latent, encoded)
        output = output * encoded['semantic']
        output = self.constrain_actions(output,encoded)
        mean = torch.tensor(self.normalization['delta']['mean'], device=device)
        std = torch.tensor(self.normalization['delta']['std'], device=device)
        observations = output[:, 1::2] * std + mean + batch['history_observations'][:, -1, None]
        observations[..., 6] = torch.atan2(observations[..., 6].sin(), observations[..., 6].cos())
        actions = output[:, ::2, :3] * torch.tensor(self.normalization['action_scale'], device=device)
        return self.consistent_output(batch,encoded,actions,observations,events,generator)
