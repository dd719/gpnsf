import math
from typing import Tuple

import torch


EPS = 1e-8


def pairwise_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""Compute pairwise Euclidean distances between 2D points.

	Args:
		a: Tensor of shape (N, D)
		b: Tensor of shape (M, D)

	Returns:
		Tensor of shape (N, M) with distances ||a_i - b_j||.
	"""
	# (N, 1, D) - (1, M, D) -> (N, M, D)
	diff = a.unsqueeze(1) - b.unsqueeze(0)
	return torch.sqrt(torch.clamp((diff**2).sum(dim=-1), min=0.0))


def exp_kernel_from_dists(dists: torch.Tensor, gamma: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
	"""Exponential kernel K = gamma * exp(- d / (2 l)).

	Args:
		dists: Distance matrix, shape (...)
		gamma: Positive scalar tensor
		lengthscale: Positive scalar tensor

	Returns:
		Kernel values with same shape as dists.
	"""
	return gamma * torch.exp(-dists / (2.0 * torch.clamp(lengthscale, min=EPS)))


def matern32_kernel_from_dists(dists: torch.Tensor, gamma: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
	"""Matérn 3/2 kernel: k(r) = gamma * (1 + sqrt(3) r / l) * exp(-sqrt(3) r / l).

	Args:
		dists: Pairwise distances (r), shape (...)
		gamma: Positive scalar tensor
		lengthscale: Positive scalar tensor

	Returns:
		Kernel values with same shape as dists.
	"""
	l = torch.clamp(lengthscale, min=EPS)
	s3 = math.sqrt(3.0)
	r = dists
	return gamma * (1.0 + s3 * r / l) * torch.exp(-s3 * r / l)


def rbf_kernel_from_dists(dists: torch.Tensor, gamma: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
	"""RBF (squared exponential) kernel: k(r) = gamma * exp(-(r^2) / (2 l^2)).

	Args:
		dists: Pairwise distances (r), shape (...)
		gamma: Positive scalar tensor
		lengthscale: Positive scalar tensor

	Returns:
		Kernel values with same shape as dists.
	"""
	l2 = torch.clamp(lengthscale, min=EPS) ** 2
	r2 = dists ** 2
	return gamma * torch.exp(-r2 / (2.0 * l2))


def softplus_pos(x: torch.Tensor) -> torch.Tensor:
	"""Numerically stable positive transform."""
	return torch.nn.functional.softplus(x) + EPS


def normalize_columns_pos(raw_w: torch.Tensor) -> torch.Tensor:
	"""Make weights positive via softplus then normalize each column to sum to 1.

	Args:
		raw_w: Tensor (K, F)
	Returns:
		Normalized and positive tensor (K, F) with column sums = 1.
	"""
	w_pos = softplus_pos(raw_w)
	col_sum = w_pos.sum(dim=0, keepdim=True) + EPS
	return w_pos / col_sum


def nb_log_prob(x: torch.Tensor, mu: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
	"""Negative Binomial log PMF with mean/dispersion parameterization.

	Parameterization: mean=mu > 0, dispersion r > 0 (size). Var = mu + mu^2 / r.

	log P(X=x) = lgamma(x+r) - lgamma(r) - lgamma(x+1)
				 + r * log(r/(r+mu)) + x * log(mu/(r+mu))

	Shapes:
		x: (..., F)
		mu: broadcastable to x
		r: (F,) or broadcastable to x
	Returns:
		elementwise log-prob tensor, same broadcasted shape as x
	"""
	x = x.to(mu.dtype)
	mu = torch.clamp(mu, min=EPS)
	r = torch.clamp(r, min=EPS)

	log_coeff = (
		torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
	)
	log_p = r * (torch.log(r) - torch.log(r + mu))
	log_q = x * (torch.log(mu) - torch.log(r + mu))
	return log_coeff + log_p + log_q


def bernoulli_log_prob(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
	"""Bernoulli log PMF with probability p in (0,1).

	Args:
		x: Tensor of 0/1 values, broadcastable to p
		p: Probability tensor, broadcastable to x
	Returns:
		elementwise log-prob tensor
	"""
	p = torch.clamp(p, min=EPS, max=1.0 - EPS)
	x = x.to(p.dtype)
	return x * torch.log(p) + (1.0 - x) * torch.log(1.0 - p)


def poisson_log_prob(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
	"""Poisson log PMF with mean mu>0.

	Args:
		x: Count tensor (>=0), broadcastable to mu
		mu: Mean tensor (>0), broadcastable to x
	Returns:
		Elementwise log-prob tensor.
	"""
	mu = torch.clamp(mu, min=EPS)
	x = x.to(mu.dtype)
	return x * torch.log(mu) - mu - torch.lgamma(x + 1.0)


@torch.no_grad()
def kmeans2d(points: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
	"""Simple K-means for 2D (or D-dim) points implemented in PyTorch.

	Args:
		points: (N, D)
		k: number of clusters
		iters: max iterations
		seed: random seed

	Returns:
		centers: (k, D)
	"""
	assert points.ndim == 2 and points.size(0) >= k
	device = points.device
	torch.manual_seed(seed)
	N, D = points.shape

	# kmeans++-like init: pick first randomly, then farthest points probabilistically
	centers = torch.empty((k, D), device=device, dtype=points.dtype)
	idx0 = torch.randint(0, N, (1,), device=device)
	centers[0] = points[idx0]
	closest = pairwise_euclidean(points, centers[0:1]).squeeze(1)
	for i in range(1, k):
		probs = torch.clamp(closest**2, min=EPS)
		probs = probs / probs.sum()
		idx = torch.multinomial(probs, 1)
		centers[i] = points[idx]
		d = pairwise_euclidean(points, centers[i:i+1]).squeeze(1)
		closest = torch.minimum(closest, d)

	for _ in range(iters):
		# Assign
		dists = pairwise_euclidean(points, centers)  # (N, k)
		labels = torch.argmin(dists, dim=1)
		# Update
		new_centers = torch.zeros_like(centers)
		counts = torch.zeros((k,), device=device, dtype=points.dtype)
		for c in range(k):
			mask = (labels == c)
			cnt = mask.sum()
			if cnt > 0:
				new_centers[c] = points[mask].mean(dim=0)
				counts[c] = cnt
			else:
				# Reinitialize empty cluster to a random point
				ridx = torch.randint(0, N, (1,), device=device)
				new_centers[c] = points[ridx]
				counts[c] = 1.0
		shift = torch.norm(new_centers - centers, dim=1).max()
		centers = new_centers
		if shift < 1e-6:
			break
	return centers


def make_inducing_points(S: torch.Tensor, M: int) -> torch.Tensor:
	"""Choose inducing points: if M>=n use S; else use k-means centers on S.

	Args:
		S: (n, 2) spatial coordinates
		M: number of inducing points

	Returns:
		Z: (M, 2)
	"""
	n = S.size(0)
	if M >= n:
		return S.clone()
	return kmeans2d(S, k=M)

