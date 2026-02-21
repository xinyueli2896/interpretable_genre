from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ModelOutput:
    recon_logits: torch.Tensor  # (B*M, input_dim)
    mu: torch.Tensor  # (B*M, latent_dim)
    logvar: torch.Tensor  # (B*M, latent_dim)
    concepts: torch.Tensor  # (B*M, num_concepts)
    class_logits: torch.Tensor  # (B, num_genres)
    concept_to_latent: torch.Tensor  # (B*M, latent_dim)


class VAEConceptModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_concepts: int,
        num_genres: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.concept_proj = nn.Linear(latent_dim, num_concepts)
        self.concept_to_latent = nn.Linear(num_concepts, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Linear(num_concepts, num_genres)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, rolls: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        if rolls.ndim == 5:
            batch, measures, tracks, steps, pitches = rolls.shape
            flat = rolls.view(batch * measures, tracks * steps * pitches)
        else:
            batch, measures, steps, pitches = rolls.shape
            flat = rolls.view(batch * measures, steps * pitches)
        hidden = self.encoder(flat)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu, logvar)

        concepts = self.concept_proj(z)
        concept_latent = self.concept_to_latent(concepts)
        recon_logits = self.decoder(z)

        concepts_reshaped = concepts.view(batch, measures, -1)
        mask_expanded = mask.unsqueeze(-1)
        masked = concepts_reshaped * mask_expanded
        denom = mask_expanded.sum(dim=1).clamp_min(1.0)
        agg = masked.sum(dim=1) / denom
        class_logits = self.classifier(agg)

        return ModelOutput(
            recon_logits=recon_logits,
            mu=mu,
            logvar=logvar,
            concepts=concepts,
            class_logits=class_logits,
            concept_to_latent=concept_latent,
        )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
