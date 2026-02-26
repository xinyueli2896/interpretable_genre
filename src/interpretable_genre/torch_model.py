import torch
from torch import nn
import torch.nn.functional as F


class ModelOutput:
    def __init__(
        self,
        recon_program_logits,
        recon_pitchdur_logits,
        mu,
        logvar,
        z,
        z_concept,
        z_residual,
        concepts,
        concept_to_latent,
        class_logits,
    ):
        self.recon_program_logits = recon_program_logits
        self.recon_pitchdur_logits = recon_pitchdur_logits
        self.mu = mu
        self.logvar = logvar
        self.z = z
        self.z_concept = z_concept
        self.z_residual = z_residual
        self.concepts = concepts
        self.concept_to_latent = concept_to_latent
        self.class_logits = class_logits


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEConceptModel(nn.Module):
    """
    Keeps training script unchanged by returning:
      recon_program_logits: (B*M, T, program_vocab)
      recon_pitchdur_logits: (B*M, T, pitchdur_vocab)

    where T = steps_per_measure * max_polyphony

    Also: splits latent into concept/residual and uses embeddings to avoid NaNs.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_concepts,
        num_genres,
        hidden_dim,
        steps_per_measure,
        max_polyphony,
        program_vocab,
        pitchdur_vocab,
        token_embed_dim,
    ):
        super().__init__()

        self.steps_per_measure = steps_per_measure
        self.max_polyphony = max_polyphony
        self.tokens_per_measure = steps_per_measure * max_polyphony

        self.program_vocab = program_vocab
        self.pitchdur_vocab = pitchdur_vocab
        self.token_embed_dim = token_embed_dim

        self.latent_dim = latent_dim
        self.z_concept_dim = latent_dim // 2
        self.z_residual_dim = latent_dim - self.z_concept_dim

        # --- Token embeddings (critical for stability) ---
        self.program_emb = nn.Embedding(program_vocab, token_embed_dim)
        self.pitchdur_emb = nn.Embedding(pitchdur_vocab, token_embed_dim)

        enc_in_dim = self.tokens_per_measure * (2 * token_embed_dim)

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Concept pathway ---
        self.latent_to_concept = nn.Linear(self.z_concept_dim, num_concepts)
        self.concept_to_latent = nn.Linear(num_concepts, self.z_concept_dim)

        # --- Classifier on predicted concepts (song-level via mean over measures) ---
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_genres),
        )

        # --- Decoder trunk ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output logits PER TOKEN (T) not per measure
        self.program_head = nn.Linear(hidden_dim, self.tokens_per_measure * program_vocab)
        self.pitchdur_head = nn.Linear(hidden_dim, self.tokens_per_measure * pitchdur_vocab)

    def forward(self, rolls, mask):
        """
        rolls: (B, M, S, P, 2) where last dim is [program_id, pitchdur_id]
        mask:  (B, M)
        """
        B, M, S, P, C = rolls.shape
        assert C == 2, f"Expected last dim=2 (program,pitchdur), got {C}"
        assert S == self.steps_per_measure, f"S mismatch: got {S}, expected {self.steps_per_measure}"
        assert P == self.max_polyphony, f"P mismatch: got {P}, expected {self.max_polyphony}"

        # Flatten measures
        rolls_flat = rolls.reshape(B * M, S * P, 2)  # (BM, T, 2)
        program_ids = rolls_flat[..., 0].clamp_min(0).clamp_max(self.program_vocab - 1)
        pitchdur_ids = rolls_flat[..., 1].clamp_min(0).clamp_max(self.pitchdur_vocab - 1)

        # Embed tokens
        prog_e = self.program_emb(program_ids)       # (BM, T, E)
        pit_e = self.pitchdur_emb(pitchdur_ids)      # (BM, T, E)
        x = torch.cat([prog_e, pit_e], dim=-1)       # (BM, T, 2E)
        x = x.reshape(B * M, -1)                     # (BM, T*2E)

        h = self.encoder(x)

        mu = self.mu(h)                              # (BM, latent)
        logvar = self.logvar(h)                      # (BM, latent)

        # Clamp logvar to prevent exp overflow -> NaNs
        logvar = logvar.clamp(-10.0, 10.0)

        z = reparameterize(mu, logvar)               # (BM, latent)

        # Split latent
        z_concept = z[:, : self.z_concept_dim]       # (BM, zc)
        z_residual = z[:, self.z_concept_dim :]      # (BM, zr)

        # Concepts from z_concept
        pred_concepts = self.latent_to_concept(z_concept)  # (BM, num_concepts)

        # Concept -> latent (concept subspace only)
        concept_latent = self.concept_to_latent(torch.sigmoid(pred_concepts))  # (BM, zc)

        # Decode full latent
        dec_h = self.decoder(z)                      # (BM, hidden)

        program_logits = self.program_head(dec_h).view(B * M, self.tokens_per_measure, self.program_vocab)
        pitchdur_logits = self.pitchdur_head(dec_h).view(B * M, self.tokens_per_measure, self.pitchdur_vocab)

        # Reshape for outputs expected by rest of code
        mu_bm = mu.view(B, M, -1)
        logvar_bm = logvar.view(B, M, -1)
        z_bm = z.view(B, M, -1)
        zc_bm = z_concept.view(B, M, -1)
        zr_bm = z_residual.view(B, M, -1)
        concepts_bm = pred_concepts.view(B, M, -1)
        concept_latent_bm = concept_latent.view(B, M, -1)

        # Song-level classifier: mean predicted concepts over measures
        class_logits = self.classifier(concepts_bm.mean(dim=1))  # (B, num_genres)

        return ModelOutput(
            recon_program_logits=program_logits,      # (BM, T, program_vocab) ✅ matches training metrics
            recon_pitchdur_logits=pitchdur_logits,    # (BM, T, pitchdur_vocab) ✅
            mu=mu_bm,
            logvar=logvar_bm,
            z=z_bm,
            z_concept=zc_bm,
            z_residual=zr_bm,
            concepts=concepts_bm,                     # (B, M, num_concepts)
            concept_to_latent=concept_latent_bm,      # (B, M, zc)
            class_logits=class_logits,                # (B, num_genres)
        )


def kl_divergence(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()