from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm

from .utils import (
	pairwise_euclidean,
	exp_kernel_from_dists,
	matern32_kernel_from_dists,
	rbf_kernel_from_dists,
	softplus_pos,
	normalize_columns_pos,
	nb_log_prob,
	bernoulli_log_prob,
	poisson_log_prob,
	make_inducing_points,
	EPS,
)


class GPNSFModel(nn.Module):
	"""GP-driven Nonnegative Shared Factorization for two modalities.

	- Shared latent factors H_i,k ~ GP_k(s_i) with mean mu_k(s) = beta0_k + s^T beta1_k
	- Kernel: K_k(s_i, s_j) = gamma_k * exp(- ||s_i - s_j|| / (2 l_k))
	- Link: y^{(r)}_{ij} = sum_k exp(h_{ik}) * w^{(r)}_{kj}, with column-normalized W^{(r)}
	- Likelihoods (with corrections provided by user):
		X1_{ij} | y1_{ij} ~ NB(mean=y1_{ij}, dispersion=theta1_j)
		X2_{ij} | y2_{ij} ~ NB(mean=y2_{ij}, dispersion=lambda2_j)

	Variational family (sparse GP): q(U,H) = p(H|U) q(U), with q(U) diagonal-cov Gaussians.
	"""

	def __init__(
		self,
		S: torch.Tensor,
		p: int,
		q: int,
		K: int,
		M: Optional[int] = None,
		eta: float = 1.0,
		num_mc_samples: int = 1,
		jitter: float = 1e-4,
		omega_type: str = "diag",
		kernel_type: str = "matern32",
		likelihood_x2: str = "nb",
		device: Optional[torch.device] = None,
		dtype: torch.dtype = torch.float32,
	) -> None:
		"""Args:
			S: (n, 2) spatial coordinates tensor
			p: features in modality 1
			q: features in modality 2
			K: number of latent factors
			M: number of inducing points (default: n)
			eta: weight on KL term in ELBO
			num_mc_samples: Monte Carlo samples for E_q[log p(X|H)]
			jitter: added to kernel diagonal for numerical stability
		"""
		super().__init__()
		if device is None:
			device = S.device
		self.device = device
		self.dtype = dtype

		self.S = S.to(device=device, dtype=dtype)  # (n, 2)
		self.n = S.size(0)
		self.p = p
		self.q = q
		self.K = K
		self.M = int(M) if M is not None else self.n
		self.eta = float(eta)
		self.num_mc_samples = int(num_mc_samples)
		self.jitter = float(jitter)
		assert likelihood_x2 in {"nb", "ber", "poi"}, "likelihood_x2 must be 'nb', 'ber' or 'poi'"
		assert omega_type in {"diag", "chol"}, "omega_type must be 'diag' or 'chol'"
		assert kernel_type in {"exp", "rbf", "matern32"}, "kernel_type must be one of 'exp', 'rbf', 'matern32'"
		self.likelihood_x2 = likelihood_x2
		self.omega_type = omega_type
		self.kernel_type = kernel_type

		# Inducing locations Z (M, 2)
		Z = make_inducing_points(self.S, self.M)
		self.register_buffer("Z", Z.to(device=device, dtype=dtype))

		# GP mean function parameters per factor: beta0 (K,), beta1 (K,2)
		self.beta0 = nn.Parameter(torch.zeros(K, device=device, dtype=dtype))
		self.beta1 = nn.Parameter(torch.zeros(K, 2, device=device, dtype=dtype))

		# Kernel hyperparameters per factor (positivity via softplus)
		self._log_gamma = nn.Parameter(torch.zeros(K, device=device, dtype=dtype))
		self._log_lengthscale = nn.Parameter(torch.zeros(K, device=device, dtype=dtype))

		# Variational parameters for inducing values per factor
		# q(u_k) = N(delta_k, Omega_k)
		self.delta = nn.Parameter(torch.zeros(K, self.M, device=device, dtype=dtype))
		# Omega_k parameterization:
		# - if omega_type == 'diag': diagonal covariance via softplus
		# - if omega_type == 'chol': full covariance via lower-triangular L_k with Ω_k = L_k L_k^T
		if self.omega_type == "diag":
			self._omega_diag_raw = nn.Parameter(torch.full((K, self.M), -2.0, device=device, dtype=dtype))
		else:
			# Initialize small values to keep initial Ω close to diagonal stability
			self._omega_tril_raw = nn.Parameter(torch.randn(K, self.M, self.M, device=device, dtype=dtype) * 1e-3)

		# Nonnegative factor loadings (raw); we enforce positivity+column normalization in forward
		self.W1_raw = nn.Parameter(torch.randn(K, p, device=device, dtype=dtype) * 0.01)
		self.W2_raw = nn.Parameter(torch.randn(K, q, device=device, dtype=dtype) * 0.01)

		# Dispersions per feature for NB in each modality (>0 via softplus)
		self.theta1_raw = nn.Parameter(torch.zeros(p, device=device, dtype=dtype))
		# For X2: separate parameters for NB, Bernoulli, and Poisson modes
		# NB dispersion (>0 via softplus)
		self.lambda2_nb_raw = nn.Parameter(torch.zeros(q, device=device, dtype=dtype))
		# Bernoulli scaling in (0,1) via sigmoid
		self.lambda2_ber_raw = nn.Parameter(torch.zeros(q, device=device, dtype=dtype))
		# Poisson scaling (>0 via softplus)
		self.lambda2_poi_raw = nn.Parameter(torch.zeros(q, device=device, dtype=dtype))

		# Precompute pairwise distances (reused across factors)
		# Dzz: (M, M), Dzh: (M, n)
		self.register_buffer("Dzz", pairwise_euclidean(self.Z, self.Z))
		self.register_buffer("Dzh", pairwise_euclidean(self.Z, self.S))

	# ---------- Helper getters ----------
	@property
	def gamma(self) -> torch.Tensor:
		return softplus_pos(self._log_gamma)

	@property
	def lengthscale(self) -> torch.Tensor:
		return softplus_pos(self._log_lengthscale)

	@property
	def theta1(self) -> torch.Tensor:
		return softplus_pos(self.theta1_raw)

	@property
	def lambda2_nb(self) -> torch.Tensor:
		return softplus_pos(self.lambda2_nb_raw)

	@property
	def lambda2_ber(self) -> torch.Tensor:
		return torch.sigmoid(self.lambda2_ber_raw)

	@property
	def lambda2_poi(self) -> torch.Tensor:
		return softplus_pos(self.lambda2_poi_raw)

	def mean_function(self, S: torch.Tensor) -> torch.Tensor:
		"""Compute mu_k(S) for all k -> shape (n, K).
		mu_k(s_i) = beta0_k + s_i^T beta1_k
		"""
		n = S.size(0)
		# S @ beta1^T -> (n, K)
		trend = S @ self.beta1.transpose(0, 1)  # (n, K)
		return trend + self.beta0.unsqueeze(0)  # (n, K)

	def mean_function_Z(self) -> torch.Tensor:
		return self.mean_function(self.Z)  # (M, K)

	# ---------- Kernel matrices per factor ----------
	def kernel_mats(self, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Compute K_uu (M,M), K_uh (M,n), and diag K_hh (n,) for factor k."""
		gk = self.gamma[k]
		lk = self.lengthscale[k]
		# Select kernel function
		if self.kernel_type == "exp":
			Kuu = exp_kernel_from_dists(self.Dzz, gk, lk)
			Kuh = exp_kernel_from_dists(self.Dzh, gk, lk)
		elif self.kernel_type == "rbf":
			Kuu = rbf_kernel_from_dists(self.Dzz, gk, lk)
			Kuh = rbf_kernel_from_dists(self.Dzh, gk, lk)
		else:  # "matern32"
			Kuu = matern32_kernel_from_dists(self.Dzz, gk, lk)
			Kuh = matern32_kernel_from_dists(self.Dzh, gk, lk)
		Kuu = Kuu + torch.eye(self.M, device=self.device, dtype=self.dtype) * self.jitter
		# For this kernel, K(s_i, s_i) = gamma
		diag_Khh = torch.full((self.n,), float(gk.item()), device=self.device, dtype=self.dtype)
		return Kuu, Kuh, diag_Khh

	# ---------- Numerical stabilization ----------
	def _symmetrize(self, A: torch.Tensor) -> torch.Tensor:
		return 0.5 * (A + A.transpose(-1, -2))

	def _robust_cholesky(self, A: torch.Tensor) -> torch.Tensor:
		"""Robust Cholesky with escalating jitter if needed.

		Attempts up to 6 times with 10x jitter escalation.
		"""
		A = self._symmetrize(A)
		jitter = self.jitter
		I = torch.eye(A.size(0), device=self.device, dtype=self.dtype)
		for _ in range(6):
			try:
				return torch.linalg.cholesky(A + I * jitter)
			except Exception:
				jitter *= 10.0
		# Final attempt with larger additive jitter based on diagonal scale
		diag_mean = torch.diag(A).abs().mean().item()
		extra = max(diag_mean, 1.0) * 1e-6
		return torch.linalg.cholesky(A + I * (jitter + extra))

	# ---------- Variational marginal q(h_k) ----------
	def q_h_params(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return (mu_tilde_k (n,), var_tilde_k (n,)) for factor k.

		mu_tilde = mu(S) + K_uh^T K_uu^{-1} (delta - mu(Z))
		var_tilde_i = K(s_i,s_i) - alpha(s_i)^T (K_uu - Omega) alpha(s_i)
			where alpha(s_i) = K_uu^{-1} K_uh[:, i]
		"""
		Kuu, Kuh, diag_Khh = self.kernel_mats(k)
		mu_S_k = self.mean_function(self.S)[:, k]  # (n,)
		mu_Z_k = self.mean_function_Z()[:, k]      # (M,)
		delta_k = self.delta[k]                     # (M,)
		# Build Omega or its factor depending on omega_type
		if self.omega_type == "diag":
			omega_diag_k = softplus_pos(self._omega_diag_raw[k])  # (M,)
		else:
			# Lower-triangular factor with positive diagonal via softplus
			L_omega = torch.tril(self._omega_tril_raw[k])  # (M, M)
			# enforce positive diagonal
			diag = softplus_pos(torch.diagonal(L_omega))
			L_omega = L_omega - torch.diag_embed(torch.diagonal(L_omega)) + torch.diag_embed(diag)

		# Cholesky of Kuu
		L = self._robust_cholesky(Kuu)

		# v = Kuu^{-1} (delta - muZ)
		rhs = (delta_k - mu_Z_k).unsqueeze(-1)  # (M,1)
		v = torch.cholesky_solve(rhs, L)  # (M,1)

		mu_tilde = mu_S_k + (Kuh.transpose(0, 1) @ v).squeeze(-1)  # (n,)

		# alpha = Kuu^{-1} Kuh ; using cholesky_solve for all columns
		alpha = torch.cholesky_solve(Kuh, L)  # (M, n)

		# Compute diagonal variance term for q(h):
		# var_diag = diag_Khh - diag(alpha^T (Kuu - Ω) alpha)
		if self.omega_type == "diag":
			B_alpha = (Kuu - torch.diag(omega_diag_k)) @ alpha  # (M, n)
			quad_diag = (alpha * B_alpha).sum(dim=0)  # (n,)
		else:
			# Avoid forming Ω explicitly: Ω alpha = L (L^T alpha)
			t1 = L_omega.transpose(0, 1) @ alpha  # (M, n)
			Omega_alpha = L_omega @ t1              # (M, n)
			B_alpha = (Kuu @ alpha) - Omega_alpha    # (M, n)
			quad_diag = (alpha * B_alpha).sum(dim=0) # (n,)
		var_tilde = diag_Khh - quad_diag
		var_tilde = torch.clamp(var_tilde, min=EPS)
		return mu_tilde, var_tilde

	def sample_H(self) -> torch.Tensor:
		"""Sample H ~ q(H) using per-row independence across factors.

		Returns:
			H_samples: (S, n, K) where S=self.num_mc_samples
		"""
		mu_list = []
		var_list = []
		for k in range(self.K):
			mu_k, var_k = self.q_h_params(k)
			mu_list.append(mu_k)
			var_list.append(var_k)
		mu = torch.stack(mu_list, dim=1)   # (n, K)
		var = torch.stack(var_list, dim=1) # (n, K)

		S = self.num_mc_samples
		if S <= 0:
			S = 1
		eps = torch.randn(S, self.n, self.K, device=self.device, dtype=self.dtype)
		H = mu.unsqueeze(0) + eps * torch.sqrt(var.unsqueeze(0))
		return H  # (S, n, K)

	def compute_KL_u(self) -> torch.Tensor:
		"""Compute sum_k KL(q(u_k) || p(u_k)).

		q(u_k) = N(delta_k, Omega_k), p(u_k)=N(mu(Z), Kuu)
		Handles diagonal (omega_type='diag') or full (omega_type='chol') Ω_k.
		"""
		total_kl = torch.zeros((), device=self.device, dtype=self.dtype)
		I = torch.eye(self.M, device=self.device, dtype=self.dtype)

		mu_Z = self.mean_function_Z()  # (M, K)
		for k in range(self.K):
			Kuu, _, _ = self.kernel_mats(k)
			L = self._robust_cholesky(Kuu)
			delta_k = self.delta[k]  # (M,)
			if self.omega_type == "diag":
				omega_diag_k = softplus_pos(self._omega_diag_raw[k])  # (M,)
			else:
				L_omega = torch.tril(self._omega_tril_raw[k])  # (M, M)
				# positive diagonal
				diag = softplus_pos(torch.diagonal(L_omega))
				L_omega = L_omega - torch.diag_embed(torch.diagonal(L_omega)) + torch.diag_embed(diag)
			muZ_k = mu_Z[:, k]

			# log|Kuu|
			logdet_Kuu = 2.0 * torch.log(torch.diag(L)).sum()
			# log|Omega|
			if self.omega_type == "diag":
				logdet_Omega = torch.log(omega_diag_k).sum()
				# trace(Kuu^{-1} Ω) via cholesky_solve on diagonal matrix
				Omega_mat = torch.diag(omega_diag_k)
				Kinv_Omega = torch.cholesky_solve(Omega_mat, L)
				trace_term = torch.diag(Kinv_Omega).sum()
			else:
				# Ω = L_omega L_omega^T, log|Ω| = 2 * sum(log diag(L_omega))
				logdet_Omega = 2.0 * torch.log(torch.diag(L_omega)).sum()
				# trace(Kuu^{-1} Ω) = sum((Kuu^{-1} L_omega) * L_omega)
				Kinv_L = torch.cholesky_solve(L_omega, L)  # (M, M)
				trace_term = (Kinv_L * L_omega).sum()

			# (delta - muZ)^T Kuu^{-1} (delta - muZ)
			rhs = (delta_k - muZ_k).unsqueeze(-1)
			v = torch.cholesky_solve(rhs, L)
			quad = (rhs.squeeze(-1) * v.squeeze(-1)).sum()

			Mdim = float(self.M)
			kl_k = 0.5 * (logdet_Kuu - logdet_Omega - Mdim + trace_term + quad)
			total_kl = total_kl + kl_k
		return total_kl

	def get_W1_W2(self) -> Tuple[torch.Tensor, torch.Tensor]:
		W1 = normalize_columns_pos(self.W1_raw)
		W2 = normalize_columns_pos(self.W2_raw)
		return W1, W2

	def elbo(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
		"""Compute ELBO = E_q[log p(X|H)] - eta * sum KL.

		Args:
			X1: (n, p)
			X2: (n, q)
		Returns:
			Scalar tensor: ELBO
		"""
		X1 = X1.to(device=self.device, dtype=self.dtype)
		X2 = X2.to(device=self.device, dtype=self.dtype)

		W1, W2 = self.get_W1_W2()
		theta1 = self.theta1  # (p,)

		Hs = self.sample_H()  # (S, n, K)
		Y1_list = []
		Y2_list = []
		for t in range(Hs.size(0)):
			H = Hs[t]
			Y1 = torch.exp(H) @ W1  # (n, p)
			Y2 = torch.exp(H) @ W2  # (n, q)
			Y1_list.append(Y1)
			Y2_list.append(Y2)

		# Monte Carlo average of log likelihoods
		loglik1 = torch.stack([
			nb_log_prob(X1, Y1, theta1).sum()
			for Y1 in Y1_list
		]).mean()
		# X2 likelihood: NB (mean=Y2, dispersion=lambda2_nb) or Bernoulli (p = clamp(Y2*lambda2_ber, 0,1)) or Poisson (mean=Y2*lambda2_poi)
		if self.likelihood_x2 == "nb":
			lambda2_nb = self.lambda2_nb  # (q,)
			loglik2 = torch.stack([
				nb_log_prob(X2, Y2, lambda2_nb).sum()
				for Y2 in Y2_list
			]).mean()
		elif self.likelihood_x2 == "ber":
			lambda2_ber = self.lambda2_ber  # (q,)
			# If X2 is counts, we binarize here (presence/absence)
			X2_bin = (X2 > 0).to(self.dtype)
			loglik2 = torch.stack([
				bernoulli_log_prob(X2_bin, torch.clamp(Y2 * lambda2_ber, min=EPS, max=1.0 - EPS)).sum()
				for Y2 in Y2_list
			]).mean()
		else:  # "poi"
			lambda2_poi = self.lambda2_poi  # (q,)
			loglik2 = torch.stack([
				poisson_log_prob(X2, torch.clamp(Y2 * lambda2_poi, min=EPS)).sum()
				for Y2 in Y2_list
			]).mean()

		kl_u = self.compute_KL_u()
		elbo = loglik1 + loglik2 - self.eta * kl_u
		return elbo

	def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
		"""Return negative ELBO as loss to minimize."""
		return -self.elbo(X1, X2)

	@torch.no_grad()
	def reconstruct(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return E[Y1], E[Y2] (or probability for X2 if Bernoulli) under q(H) using mean (no noise)."""
		mu_list, var_list = [], []
		for k in range(self.K):
			mu_k, _ = self.q_h_params(k)
			mu_list.append(mu_k)
		mu = torch.stack(mu_list, dim=1)  # (n, K)
		W1, W2 = self.get_W1_W2()
		Y1 = torch.exp(mu) @ W1
		Y2_mean = torch.exp(mu) @ W2
		if self.likelihood_x2 == "nb":
			return Y1, Y2_mean
		elif self.likelihood_x2 == "ber":
			# Return probability matrix for Bernoulli
			lambda2_ber = self.lambda2_ber
			P2 = torch.clamp(Y2_mean * lambda2_ber, min=EPS, max=1.0 - EPS)
			return Y1, P2
		else:
			# Return mean for Poisson
			lambda2_poi = self.lambda2_poi
			Mu2 = torch.clamp(Y2_mean * lambda2_poi, min=EPS)
			return Y1, Mu2



def compute_loglik_terms(model, X1, X2):
	with torch.no_grad():
		W1, W2 = model.get_W1_W2()
		theta1 = model.theta1
		Hs = model.sample_H()  # (S, n, K)
		loglik1 = []
		loglik2 = []
		if model.likelihood_x2 == 'nb':
			lambda2_nb = model.lambda2_nb
		elif model.likelihood_x2 == 'bernoulli':
			lambda2_ber = model.lambda2_ber
			X2_bin = (X2 > 0).to(X2.dtype)
		else:
			lambda2_poi = model.lambda2_poi
		for t in range(Hs.size(0)):
			H = Hs[t]
			Y1 = torch.exp(H) @ W1
			Y2 = torch.exp(H) @ W2
			loglik1.append(nb_log_prob(X1, Y1, theta1).sum())
			if model.likelihood_x2 == 'nb':
				loglik2.append(nb_log_prob(X2, Y2, lambda2_nb).sum())
			elif model.likelihood_x2 == 'ber':
				p2 = torch.clamp(Y2 * lambda2_ber, min=1e-8, max=1.0 - 1e-8)
				loglik2.append(bernoulli_log_prob(X2_bin, p2).sum())
			else:    
				loglik2.append(poisson_log_prob(X2, Y2*lambda2_poi).sum())
		loglik1 = torch.stack(loglik1).mean()
		loglik2 = torch.stack(loglik2).mean()
	return loglik1, loglik2

def train_model(model, X1_t, X2_t, num_steps=5000, lr_schedule=None, print_every=250):
    """
    训练模型的函数
    
    参数:
        model: GPNSFModel 实例
        X1_t, X2_t: 训练数据
        num_steps: 总训练步数
        lr_schedule: 学习率调度配置，例如:
            {'initial': 1e-2, 'switch_step': 2000, 'final': 5e-4}
        print_every: 打印间隔
    """
    if lr_schedule is None:
        lr_schedule = {'initial': 1e-2, 'switch_step': 4000, 'final': 5e-4}
    
    # 设置优化器
    opt = optim.Adam(model.parameters(), lr=lr_schedule['initial'])
    
    # 训练循环
    for step in tqdm(range(1, num_steps + 1), total=num_steps):
        opt.zero_grad()
        loss = model(X1_t, X2_t)  # 负ELBO
        loss.backward()
        opt.step()

        # 学习率调度
        if step == lr_schedule.get('switch_step', 0):
            for param_group in opt.param_groups:
                param_group['lr'] = lr_schedule['final']
            print(f"\n[step {step:03d}] Learning rate changed to {lr_schedule['final']}")

        # 打印训练信息
        if step == 1 or step % print_every == 0:
            with torch.no_grad():
                elbo_val = model.elbo(X1_t, X2_t)
                kl_val = model.compute_KL_u()
                loglik1, loglik2 = compute_loglik_terms(model, X1_t, X2_t)
                current_lr = opt.param_groups[0]['lr']
                
                print(f"[step {step:03d}] loss={loss.item():.3f}  ELBO={elbo_val.item():.3f}  "
                      f"KL={kl_val.item():.3f}  loglik1={loglik1.item():.3f}  "
                      f"loglik2={loglik2.item():.3f}  lr={current_lr:.1e}")
    
    print('Training finished.')
    return model

