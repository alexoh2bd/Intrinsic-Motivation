"""
Embedding quality metrics for analyzing CRL representations.

All functions accept NumPy arrays of shape (N, D) where N is the number
of samples and D is the embedding dimension.
"""
import numpy as np
from typing import Dict


# ── Singular value helpers ──────────────────────────────────────────────────

def singular_value_spectrum(embeddings: np.ndarray) -> np.ndarray:
    """Compute singular values of the centered embedding matrix (N, D)."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return s


# ── Rank and spectrum metrics ───────────────────────────────────────────────

def numerical_rank(embeddings: np.ndarray, threshold: float = 0.01) -> int:
    """Count singular values above *threshold* × max singular value."""
    s = singular_value_spectrum(embeddings)
    return int(np.sum(s > threshold * s[0]))


def effective_rank(embeddings: np.ndarray) -> float:
    """
    Roy & Vetterli (2007): exp(Shannon entropy of the normalized singular
    value distribution).  Continuous generalization of numerical rank.
    """
    s = singular_value_spectrum(embeddings)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def condition_number(embeddings: np.ndarray) -> float:
    """Ratio of largest to smallest singular value."""
    s = singular_value_spectrum(embeddings)
    return float(s[0] / (s[-1] + 1e-10))


def participation_ratio(embeddings: np.ndarray) -> float:
    """(sum λ_i)^2 / sum(λ_i^2)  where λ_i are eigenvalues (= σ_i^2)."""
    s = singular_value_spectrum(embeddings)
    eigenvalues = s ** 2
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def pca_explained_variance(embeddings: np.ndarray) -> np.ndarray:
    """Cumulative explained variance ratio for each principal component."""
    s = singular_value_spectrum(embeddings)
    variance = s ** 2
    return np.cumsum(variance / variance.sum())


# ── Intrinsic dimensionality ───────────────────────────────────────────────

def two_nn_intrinsic_dimension(
    embeddings: np.ndarray, max_samples: int = 2000
) -> float:
    """
    Two-NN estimator (Facco et al., 2017).  MLE from the ratio of
    distances to the 1st and 2nd nearest neighbors.
    """
    n = len(embeddings)
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        embeddings = embeddings[idx]
        n = max_samples

    # Pairwise L2 distances
    dists = np.sqrt(
        np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1)
    )
    np.fill_diagonal(dists, np.inf)

    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    mu = r2 / (r1 + 1e-10)
    mu = mu[mu > 1.0]

    if len(mu) < 10:
        return 0.0

    return float(len(mu) / np.sum(np.log(mu)))


# ── Contrastive / goal-reaching quality ─────────────────────────────────────

def alignment(sa_embeddings: np.ndarray, g_embeddings: np.ndarray) -> float:
    """Mean L2 distance between matched (sa_i, g_i) positive pairs."""
    return float(
        np.sqrt(np.sum((sa_embeddings - g_embeddings) ** 2, axis=-1)).mean()
    )


def uniformity(
    embeddings: np.ndarray, t: float = 2.0, max_samples: int = 2000
) -> float:
    """
    log(mean exp(-t ‖x_i − x_j‖²))  (Wang & Isola, 2020).
    More negative ⇒ more uniform spread on the hypersphere.
    """
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]

    sq_dists = np.sum(
        (embeddings[:, None] - embeddings[None, :]) ** 2, axis=-1
    )
    mask = ~np.eye(len(embeddings), dtype=bool)
    return float(np.log(np.exp(-t * sq_dists[mask]).mean() + 1e-10))


def positive_negative_distance_ratio(
    sa_embeddings: np.ndarray,
    g_embeddings: np.ndarray,
    max_neg_samples: int = 1000,
) -> float:
    """
    mean(‖sa_i − g_i‖) / mean(‖sa_i − g_j‖)  for i ≠ j.
    Should be ≪ 1 for a good contrastive representation.
    """
    pos_dists = np.sqrt(
        np.sum((sa_embeddings - g_embeddings) ** 2, axis=-1)
    )
    n = min(len(sa_embeddings), max_neg_samples)
    sa_sub = sa_embeddings[:n]
    g_sub = g_embeddings[:n]

    neg_dists = np.sqrt(
        np.sum((sa_sub[:, None] - g_sub[None, :]) ** 2, axis=-1)
    )
    mask = ~np.eye(n, dtype=bool)

    return float(pos_dists.mean() / (neg_dists[mask].mean() + 1e-10))


# ── Geometry ────────────────────────────────────────────────────────────────

def isotropy_score(embeddings: np.ndarray) -> float:
    """min(eigenvalue) / max(eigenvalue) of the covariance matrix."""
    s = singular_value_spectrum(embeddings)
    eig = s ** 2
    return float(eig[-1] / (eig[0] + 1e-10))


def mean_embedding_norm(embeddings: np.ndarray) -> float:
    """Mean L2 norm of embeddings (centering diagnostic)."""
    return float(np.sqrt(np.sum(embeddings ** 2, axis=-1)).mean())


def cosine_similarity_distribution(
    embeddings: np.ndarray, max_samples: int = 2000
) -> np.ndarray:
    """Pairwise cosine similarities (upper triangle, excluding diagonal)."""
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]

    norms = np.sqrt(np.sum(embeddings ** 2, axis=-1, keepdims=True))
    normed = embeddings / (norms + 1e-10)
    sim = normed @ normed.T
    idx = np.triu_indices(len(embeddings), k=1)
    return sim[idx]


# ── Aggregate ───────────────────────────────────────────────────────────────

def compute_all_metrics(
    sa_embeddings: np.ndarray,
    g_embeddings: np.ndarray,
) -> Dict[str, float]:
    """Compute every metric for a given layer's SA and Goal embeddings."""
    all_emb = np.concatenate([sa_embeddings, g_embeddings], axis=0)

    return {
        # Rank & spectrum
        "numerical_rank_sa": numerical_rank(sa_embeddings),
        "numerical_rank_goal": numerical_rank(g_embeddings),
        "numerical_rank_joint": numerical_rank(all_emb),
        "effective_rank_sa": effective_rank(sa_embeddings),
        "effective_rank_goal": effective_rank(g_embeddings),
        "effective_rank_joint": effective_rank(all_emb),
        "participation_ratio_sa": participation_ratio(sa_embeddings),
        "participation_ratio_goal": participation_ratio(g_embeddings),
        "condition_number_sa": condition_number(sa_embeddings),
        "condition_number_goal": condition_number(g_embeddings),
        # Intrinsic dimensionality
        "two_nn_dim_sa": two_nn_intrinsic_dimension(sa_embeddings),
        "two_nn_dim_goal": two_nn_intrinsic_dimension(g_embeddings),
        "two_nn_dim_joint": two_nn_intrinsic_dimension(all_emb),
        # Contrastive quality
        "alignment": alignment(sa_embeddings, g_embeddings),
        "uniformity_sa": uniformity(sa_embeddings),
        "uniformity_goal": uniformity(g_embeddings),
        "pos_neg_ratio": positive_negative_distance_ratio(
            sa_embeddings, g_embeddings
        ),
        # Geometry
        "isotropy_sa": isotropy_score(sa_embeddings),
        "isotropy_goal": isotropy_score(g_embeddings),
        "mean_norm_sa": mean_embedding_norm(sa_embeddings),
        "mean_norm_goal": mean_embedding_norm(g_embeddings),
    }
