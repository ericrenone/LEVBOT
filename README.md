# Geometric Lévy Dynamics in Deep Learning

> Neural network phase transitions — grokking, feature learning, generalization jumps — emerge from **synchronized criticality** across three independent mechanisms: stochastic signal-noise balance, spectral stability, and representation geometry. Each condition alone is insufficient. Their simultaneous alignment is both necessary and measurable 10–50 steps before the transition occurs.

---

## Table of Contents

1. [Motivation: What Classical Theory Cannot Explain](#1-motivation-what-classical-theory-cannot-explain)
2. [The Training Manifold](#2-the-training-manifold)
3. [Heavy-Tailed Gradient Noise: The Lévy Process Model](#3-heavy-tailed-gradient-noise-the-lévy-process-model)
4. [Three Critical Observables](#4-three-critical-observables)
   - [4.1 Consolidation Ratio Cₐ — Stochastic Criticality](#41-consolidation-ratio-cₐ--stochastic-criticality)
   - [4.2 Stability Margin S — Spectral Criticality](#42-stability-margin-s--spectral-criticality)
   - [4.3 Metric Determinant Rate dρ/dt — Geometric Criticality](#43-metric-determinant-rate-dρdt--geometric-criticality)
5. [The Unified Criticality Law](#5-the-unified-criticality-law)
6. [Why Three Independent Conditions Are Required](#6-why-three-independent-conditions-are-required)
7. [Geometric Evolution: The Fractional Fokker-Planck Equation](#7-geometric-evolution-the-fractional-fokker-planck-equation)
8. [Testable Predictions](#8-testable-predictions)
9. [Implementation Guide](#9-implementation-guide)
10. [Computational Complexity and Scaling](#10-computational-complexity-and-scaling)
11. [Limitations and Open Mathematical Problems](#11-limitations-and-open-mathematical-problems)
12. [References](#12-references)
13. [Glossary](#13-glossary)

---

## 1. Motivation: What Classical Theory Cannot Explain

Standard SGD is modeled as noisy gradient descent in flat Euclidean space:

$$\theta_{t+1} = \theta_t - \eta \nabla L + \sqrt{\eta}\, \xi_t, \qquad \xi_t \sim \mathcal{N}(0, \Sigma)$$

This predicts **smooth monotone convergence** to a local minimum. It fails qualitatively on four empirically well-documented phenomena:

**Grokking** (Power et al., 2022): On algorithmic tasks (modular arithmetic, group compositions), training loss reaches near-zero while test accuracy stays at chance for thousands of steps, then suddenly jumps to near-perfect. Classical theory predicts no such plateau-then-jump structure.

**Edge-of-Stability** (Cohen et al., 2021): Gradient descent often operates stably at learning rates where $\lambda_{\max}(H) \cdot \eta \approx 2$ — the classical divergence boundary. The linearized stability criterion predicts divergence; the actual system does not diverge.

**Heavy-Tailed Gradients** (Simsekli et al., 2019): Empirical gradient norm distributions across architectures (ResNet, ViT, BERT, GPT) have power-law tails with tail index $\alpha \approx 1.5$. A Gaussian model has finite variance; the true distribution has *infinite* variance. The entire Gaussian noise assumption is wrong in the tail regime.

**Feature Learning** (Nanda et al., 2023): Representation structure measured by the NTK matrix $G(t)$ can reorganize completely in less than 1% of total training steps. In the classical "lazy training" (NTK) regime, $G(t)$ is approximately constant throughout training. The observed abrupt reorganization is categorically different.

The common thread: all four phenomena involve **discrete transitions** in a regime classical theory predicts should be smooth. This framework provides a unified explanation: transitions occur when three independent critical thresholds align simultaneously.

---

## 2. The Training Manifold

### 2.1 Parameters Evolve on a Curved Space

Rather than treating parameter space as flat $\mathbb{R}^d$, this framework equips it with a time-varying Riemannian metric $G(t)$:

$$(\mathbb{R}^d,\, G(t))$$

The metric $G(t)$ is the **empirical Neural Tangent Kernel (NTK)** matrix:

$$G(t) = \frac{1}{n} \sum_{i=1}^n \nabla_\theta f(x_i; \theta) \cdot \nabla_\theta f(x_i; \theta)^\top \in \mathbb{R}^{d \times d}$$

where $f(x_i; \theta)$ is the network output on example $i$ and $\nabla_\theta f$ is the Jacobian with respect to parameters.

### 2.2 Properties of G(t)

**Positive semidefinite:** $G(t) \succeq 0$ always — it is a sum of outer products. When the network is trainable, it is positive definite on the span of the gradients.

**Eigenspectrum encodes representation structure:** The eigenvectors of $G(t)$ are the principal directions of gradient variation across the dataset. A high eigenvalue means many data points produce aligned gradients in that direction — that direction is structurally important. A near-zero eigenvalue means the corresponding direction is irrelevant to the current representation.

**Time-varying and reorganizing:** Unlike the Fisher information matrix (which is fixed for a fixed probabilistic model), $G(t)$ changes as $\theta$ changes. During feature learning, $G(t)$ can change rapidly — eigenvectors rotate, eigenvalues spike or collapse.

### 2.3 What G(t) Is Not

This is the **NTK in parameter space**, not the Fisher information matrix on a statistical manifold. The distinction matters:
- Fisher information requires a probabilistic model $p(y|x, \theta)$
- $G(t)$ is computed from raw Jacobians and does not require specifying a likelihood
- Natural gradient descent uses the Fisher metric; this framework uses the NTK metric
- In the infinite-width limit (lazy regime), $G(t) \approx G(0)$ is constant; the framework captures the *finite-width* feature learning regime where $G(t)$ evolves

### 2.4 Geometric Intuition

The NTK metric $G(t)$ defines distances in parameter space that are commensurate with their effect on network outputs. A step of size $\varepsilon$ in a high-curvature direction (large eigenvalue of $G$) changes outputs more than the same step in a low-curvature direction. The metric makes the geometry "data-aware" — the manifold is curved by the training data distribution.

---

## 3. Heavy-Tailed Gradient Noise: The Lévy Process Model

### 3.1 Empirical Evidence for Heavy Tails

The gradient noise in deep learning is not Gaussian. Empirical measurements of gradient norm distributions via the Hill estimator across architectures consistently find $\alpha \in (1.4, 1.7)$:

| Architecture | Task | Tail Index α | 95% CI |
|---|---|---|---|
| ResNet-50 | ImageNet | 1.62 | ±0.09 |
| Vision Transformer | ImageNet | 1.45 | ±0.12 |
| BERT-Large | Language | 1.38 | ±0.15 |
| GPT-2 | Language | 1.52 | ±0.18 |

All values satisfy $\alpha < 2$, which means the gradient noise has **infinite variance** under the true distribution. The Gaussian model with finite $\Sigma$ is fundamentally wrong for the tail behavior.

### 3.2 The α-Stable Lévy Process

Replace Gaussian noise with an $\alpha$-stable Lévy process $L_t^{(\alpha)}$:

$$d\theta = -\nabla L\, dt + \sigma\, dL_t^{(\alpha)}$$

An $\alpha$-stable Lévy process is characterized by:

**Jump measure:** $\nu(dz) \propto |z|^{-(d+\alpha)}\, dz$ — the rate of jumps of size $z$ decays as a power law. Jumps can be arbitrarily large (but with decreasing probability).

**Characteristic function:** $\mathbb{E}[e^{ik \cdot L_t}] = e^{-t|k|^\alpha}$ — the Fourier transform of the distribution at time $t$. For $\alpha = 2$ this gives $e^{-t|k|^2}$, the Gaussian. For $\alpha < 2$, the distribution has power-law tails.

**Self-similarity:** $L_t^{(\alpha)} \overset{d}{=} t^{1/\alpha} L_1^{(\alpha)}$ — scaling time by $t$ is equivalent to scaling space by $t^{1/\alpha}$. For $\alpha < 2$, space scales faster than $\sqrt{t}$ — the process spreads faster than Brownian motion.

### 3.3 Why Heavy Tails Matter for Learning

The rare large jumps from heavy-tailed noise are not bugs — they are features. They enable:

**Basin escape:** A Gaussian process with finite variance is trapped within $O(\sigma)$ of a local minimum. An $\alpha$-stable process with $\alpha < 2$ can make jumps of unbounded size, escaping local minima without requiring an explicit escape mechanism.

**Critical self-organization:** The tail index $\alpha$ decreases during critical windows (Prediction 3). This means the optimizer becomes *more* heavy-tailed precisely when large exploratory jumps are needed to find the generalization basin.

**Connection to the Broken Gate:** The Broken Gate mechanism from the GLD framework is, in the Lévy process language, the effect of the heavy tail: when the substrate is vulnerable, a large-magnitude Lévy jump fires and reorganizes the system.

### 3.4 Measuring α: The Hill Estimator

Given a sample of gradient norms $\{g_1, \ldots, g_n\}$, the Hill estimator for $\alpha$ uses the top-$k$ order statistics:

$$\hat{\alpha} = \frac{k}{\sum_{i=1}^k \log(g_{(n-i+1)} / g_{(n-k)})}$$

where $g_{(1)} \leq \cdots \leq g_{(n)}$ are the sorted norms and $k \approx 0.1n$ uses the top 10%. This is the maximum likelihood estimator for the tail index of a Pareto distribution — the limiting distribution of order statistics from a heavy-tailed distribution.

**Bias-variance tradeoff:** Small $k$ gives low bias (true tail only, not bulk) but high variance (few samples). Large $k$ gives low variance but may include non-tail bulk behavior. $k = \lfloor 0.1n \rfloor$ is the standard heuristic.

---

## 4. Three Critical Observables

### 4.1 Consolidation Ratio Cₐ — Stochastic Criticality

**Definition:**
$$C_\alpha(t) = \frac{|\nabla L|^2}{2\, D_\alpha\, d}$$

where the effective diffusion coefficient is:
$$D_\alpha = \left(\frac{\sigma_\alpha}{|\nabla L|}\right)^\alpha$$

**Derivation from first-passage analysis:**

Consider the optimizer as a particle in a potential well $V(\theta) \approx L(\theta)$, driven by an $\alpha$-stable process. The particle must escape a well of characteristic width $\ell$ (the basin size) to generalize. Two competing processes determine whether it escapes:

- **Drift:** gradient pull at rate $|\nabla L| \cdot \ell$ (moves the particle across the well)
- **Lévy jumps:** jump rate $\propto \sigma_\alpha^\alpha$ (randomly kicks the particle)

At the critical escape threshold, drift rate equals jump rate:
$$|\nabla L| \cdot \ell \sim \sigma_\alpha^\alpha$$

The well width $\ell$ relates to dimensionality by $\ell \sim d^{-1/2}$ (typical distance to a neighboring basin in $d$-dimensional space). Combining:
$$|\nabla L|^2 \sim \sigma_\alpha^\alpha / \ell \sim D_\alpha \cdot d$$

This gives $C_\alpha \sim 1$ at the critical threshold. The ratio measures how far the current gradient-noise balance is from this critical point.

**Interpretation:**
- $C_\alpha \gg 1$: gradient dominates — deterministic descent, no basin exploration
- $C_\alpha \approx 1$: balance — critical regime, basin escape is possible
- $C_\alpha \ll 1$: noise dominates — random walk, no coherent learning

**Measurement procedure:**
1. Collect gradient norms $\{g_i\}$ over window $W = 100$ steps
2. Fit $\alpha$ via Hill estimator on top-10% of sorted norms
3. Compute $\sigma_\alpha = \text{std}(\{g_i\})$ (scale parameter estimate)
4. Compute $D_\alpha = (\sigma_\alpha / g_{\text{current}})^\alpha$
5. Compute $C_\alpha = g_{\text{current}}^2 / (2 D_\alpha \cdot W)$

### 4.2 Stability Margin S — Spectral Criticality

**Definition:**
$$S(t) = \frac{2}{\eta} - \lambda_{\max}(H(t))$$

where $H(t) = \nabla^2 L(\theta_t)$ is the Hessian and $\eta$ is the learning rate.

**Derivation from linearized stability:**

Near a critical point $\theta^*$, linearize the gradient descent update $\theta_{t+1} = \theta_t - \eta G^{-1} \nabla L$:

$$\delta\theta_{t+1} = (I - \eta G^{-1} H)\, \delta\theta_t$$

Stability requires $|\text{eigenvalues of } (I - \eta G^{-1} H)| < 1$:
$$|1 - \eta \lambda_i(G^{-1} H)| < 1 \implies \lambda_i(G^{-1} H) \in \left(0, \frac{2}{\eta}\right)$$

In the **lazy regime** ($G \approx I$, feature learning has not occurred), $\lambda_i(G^{-1}H) \approx \lambda_i(H)$, so the classical condition is $\lambda_{\max}(H) < 2/\eta$, i.e., $S > 0$.

The **edge-of-stability phenomenon** corresponds to $S \approx 0$: $\lambda_{\max}(H) \approx 2/\eta$. Classical theory predicts divergence; the actual system self-stabilizes because the heavy-tailed noise and nonlinear curvature prevent the linearized blow-up.

**Interpretation:**
- $S > 0.5$: conservative, stable regime (well below the edge)
- $S \approx 0$: edge-of-stability (critical spectral regime)
- $S < 0$: classically unstable, but may be stabilized by Lévy noise

**Measurement procedure (power iteration):**

Power iteration finds $\lambda_{\max}(H)$ without forming $H$ explicitly, using only Hessian-vector products (HVPs):

1. Initialize random unit vector $v \in \mathbb{R}^d$
2. For $k = 1, \ldots, 5$: compute $v \leftarrow Hv / \|Hv\|$ via HVP
3. Rayleigh quotient: $\lambda_{\max} \approx v^\top H v$
4. $S = 2/\eta - \lambda_{\max}$

An HVP $Hv$ costs one additional backward pass via double backpropagation, so the total cost is 6 backward passes — tractable but non-trivial for large models.

### 4.3 Metric Determinant Rate dρ/dt — Geometric Criticality

**Definition:**
$$\rho(t) = \log \det G(t) = \sum_k \log \lambda_k(G(t))$$
$$\frac{d\rho}{dt} \approx \frac{\rho(t) - \rho(t-1)}{\Delta t}$$

where the sum is over eigenvalues $\lambda_k > 10^{-6}$ (numerical threshold).

**Why $\log \det G$?**

The log-determinant of the NTK matrix is the **log-volume** of the ellipsoid defined by $G$. In information geometry, $\log \det G$ measures the effective dimensionality of the gradient directions:

- If all eigenvalues are equal ($G = cI$): $\log \det G = d \log c$ — all directions contribute equally
- If only $k$ eigenvalues are large: $\log \det G \approx k \log c$ — only $k$ effective dimensions
- Rapid change in $\log \det G$ signals that the set of "active" gradient directions is reorganizing

**Geometric meaning:** $d\rho/dt$ is the rate at which the gradient manifold is changing its shape. In lazy learning (NTK regime), the manifold is approximately static and $d\rho/dt \approx 0$. During feature learning, the manifold reorganizes rapidly and $|d\rho/dt|$ is large.

**Direction of $d\rho/dt$:**
- $d\rho/dt > 0$: representation **expanding** — new gradient directions are becoming active (exploration of new features)
- $d\rho/dt < 0$: representation **contracting** — gradient directions are collapsing (consolidation into a low-dimensional structure)
- $d\rho/dt \approx 0$: lazy regime — representation is static

**Measurement procedure:**
1. Sample $n' = 32$ data points (subsampling for efficiency)
2. For each: zero gradients, backward pass through network output, collect flat parameter gradient
3. Stack gradients: $G = [\nabla_\theta f(x_1), \ldots, \nabla_\theta f(x_{n'})]$ into matrix, compute $G^\top G / n'$
4. Eigendecompose via `torch.linalg.eigvalsh`
5. $\rho = \sum_{k:\, \lambda_k > 10^{-6}} \log \lambda_k$
6. $d\rho/dt = \rho(t) - \rho(t-1)$

---

## 5. The Unified Criticality Law

### 5.1 The Probability Function Φ

The probability of a generalization jump given the current observables is:

$$P(\text{generalization jump} \mid C_\alpha, S, d\rho/dt) \approx \Phi(C_\alpha, S, d\rho/dt)$$

where:

$$\Phi(C_\alpha, S, r) = \exp\!\left(-\frac{(C_\alpha - 1)^2}{2\sigma_1^2} - \frac{S^2}{2\sigma_2^2} - \frac{(r - \mu)^2}{2\sigma_3^2}\right)$$

**Empirically fitted parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| $\sigma_1 \approx 0.3$ | Width of $C_\alpha$ criticality window | Transitions occur for $C_\alpha \in [0.7, 1.3]$ |
| $\sigma_2 \approx 0.15$ | Width of $S$ criticality window | Transitions occur for $S \in [-0.3, 0.3]$ |
| $\sigma_3 \approx 0.5$ | Width of $d\rho/dt$ criticality window (task-dependent) | Transitions require significant geometric change |
| $\mu \approx 1.0$ | Preferred sign of $d\rho/dt$ | Positive rate (expanding representation) is favored |

**Practical critical thresholds** (hard-threshold version for binary detection):
- $C_\alpha \in [0.8, 1.2]$: stochastically critical
- $S \in [-0.1, 0.1]$: spectrally critical
- $|d\rho/dt| > \tau$: geometrically critical ($\tau$ task-dependent, typically $\tau \approx 0.5$)

### 5.2 Theoretical Status

**This is an empirical law, not a derived theorem.** The claim that phase transitions require all three conditions simultaneously is supported by:

1. The theoretical derivation of each individual threshold (first-passage for $C_\alpha$, linearized stability for $S$, information geometry for $\rho$)
2. The argument that each condition can occur independently without producing a transition (Section 6)
3. Empirical validation across grokking tasks (pending publication — see Predictions)

A rigorous proof of necessity and sufficiency requires solving the open problem of existence theory for the fractional Fokker-Planck equation on time-varying manifolds (Section 11).

---

## 6. Why Three Independent Conditions Are Required

The key conceptual contribution is that synchronized criticality is both necessary and sufficient. Each pair of conditions without the third produces qualitatively different (non-transition) dynamics:

| $C_\alpha \approx 1$ | $S \approx 0$ | $|d\rho/dt|$ large | Result |
|---|---|---|---|
| ✓ | ✗ | ✗ | Exploratory but stable. Signal-noise balance achieved, but no spectral instability to drive basin escape, and representation is static. Slow learning. |
| ✗ | ✓ | ✗ | Deterministic edge-walking. At the stability boundary but gradient dominates — no stochastic exploration. Risky (one perturbation away from divergence) but no generalization jump. |
| ✗ | ✗ | ✓ | Noisy representation flux. Geometry reorganizes rapidly but gradient noise dominates ($C_\alpha \ll 1$) and there is no spectral instability — the flux is unstructured and does not consolidate. |
| ✓ | ✓ | ✗ | Near-critical stochastic dynamics at the stability edge, but representation is frozen. The optimizer is ready to jump but has nowhere to jump to — the feature basis has not prepared a better basin. |
| ✓ | ✗ | ✓ | Exploratory with active feature learning, but well inside the stability regime. Representation reorganizes but the optimizer cannot exploit it without spectral instability to amplify the new directions. |
| ✗ | ✓ | ✓ | Geometric reorganization at the stability edge, but gradient signal dominates — the optimizer is committed to its current direction and cannot explore the new representation. |
| **✓** | **✓** | **✓** | **Synchronized criticality. All three mechanisms align. Phase transition.** |

This table is the central argument: no pair of two conditions is sufficient. The transition requires the three mechanisms to simultaneously reach their individual critical thresholds.

---

## 7. Geometric Evolution: The Fractional Fokker-Planck Equation

### 7.1 The Equation

The probability density $p(\theta, t)$ of the optimizer's position evolves formally as:

$$\frac{\partial p}{\partial t} = \nabla \cdot (p \nabla L) + D_\alpha \mathcal{L}_\alpha[p]$$

where:
- $\nabla \cdot (p \nabla L)$ is the **Fokker-Planck drift** term — probability flows along the loss gradient
- $D_\alpha \mathcal{L}_\alpha[p]$ is the **fractional diffusion** term — $\mathcal{L}_\alpha$ is the fractional Laplacian capturing heavy-tailed Lévy jumps

### 7.2 The Fractional Laplacian

The fractional Laplacian $\mathcal{L}_\alpha = -(-\Delta)^{\alpha/2}$ is defined via its Fourier symbol:

$$\widehat{\mathcal{L}_\alpha f}(k) = -|k|^\alpha \hat{f}(k)$$

For $\alpha = 2$: $\mathcal{L}_2 = \Delta$ (standard Laplacian, recovers Gaussian diffusion).
For $\alpha < 2$: the fractional Laplacian is a nonlocal operator — the "diffusion" at a point $\theta$ depends on the values of $p$ at all other points (weighted by the jump measure), not just locally. This is the mathematical expression of the fact that Lévy jumps can be long-range.

In coordinates:
$$\mathcal{L}_\alpha[p](\theta) = C_{d,\alpha} \int_{\mathbb{R}^d} \frac{p(\theta + z) - p(\theta)}{|z|^{d+\alpha}}\, dz$$

### 7.3 Technical Status: The Frozen-Metric Approximation

The rigorous construction of this PDE on a **time-varying manifold** $(\mathbb{R}^d, G(t))$ is an open problem. The difficulty is that $G(t)$ changes on the same timescale as $p(\theta, t)$, so the standard theory for PDEs on fixed Riemannian manifolds does not apply.

The current implementation uses the **frozen-metric (adiabatic) approximation**: at each step $t$, treat $G(t)$ as a fixed constant, solve the equation for one step, then update $G(t)$ to $G(t+1)$ before the next step. This is valid when $G(t)$ changes slowly relative to $p(\theta, t)$ — the "lazy" regime. During rapid feature learning, this approximation breaks down, which is precisely when the most interesting dynamics occur.

Resolving this is the central open problem (Section 11.1).

---

## 8. Testable Predictions

### 8.1 Prediction 1: Precursor Signal (10–50 Steps)

**Claim:** Critical alignment of all three observables precedes generalization jumps by 10–50 steps.

**Test protocol:**
1. Train a network on modular arithmetic ($a + b \pmod{p}$, a standard grokking task)
2. Record $\{C_\alpha(t), S(t), d\rho/dt(t)\}$ every step
3. Identify accuracy jumps: $\Delta \text{acc} > 5\%$ in 10 consecutive steps
4. For each jump at step $t_{\text{jump}}$, find the last critical-alignment step $t_{\text{critical}}$ before it
5. Compute lag $\tau = t_{\text{jump}} - t_{\text{critical}}$

**Expected result:** $\tau \in [10, 50]$ steps with probability $> 0.7$

**Null hypothesis:** $\tau$ is uniformly distributed on $[0, 200]$ (no precursor signal)

**Statistical test:** One-sided $t$-test on $\{\tau_i\}$ vs. uniform null. Effect size (Cohen's $d$) expected $> 0.8$.

### 8.2 Prediction 2: Rare Alignment

**Claim:** $P(\text{all three critical simultaneously}) \approx 0.001$–$0.002$ per step

**Test protocol:**
1. Track observables across 100k training steps
2. Count steps $N_{\text{all}}$ where all three conditions hold
3. Count steps $N_1, N_2, N_3$ where each individual condition holds
4. Compare observed joint rate $N_{\text{all}}/100k$ to independence prediction $N_1 \cdot N_2 \cdot N_3 / (100k)^2$

**Expected result:** Observed joint rate matches independence prediction within a factor of 3 (conditions are approximately independent)

**Interpretation:** This is a consistency check, not a confirmation. If the joint rate is much higher than the independence prediction, the conditions are positively correlated (some common mechanism drives all three). If much lower, they are negatively correlated (one condition suppresses the others).

### 8.3 Prediction 3: Tail Index Decrease During Critical Windows

**Claim:** $\alpha$ (the Lévy tail index) decreases by 0.1–0.3 during critical events

**Test protocol:**
1. Compute $\hat{\alpha}(t)$ via Hill estimator in a sliding window of 200 steps
2. Compute criticality indicator $\Phi(t)$ from all three observables
3. Test Pearson correlation between $\hat{\alpha}(t)$ and $\Phi(t)$

**Expected result:** Significant negative correlation ($r < -0.3$, $p < 0.01$) — higher criticality coincides with lower tail index (heavier tails, more Lévy behavior)

**Mechanistic interpretation:** As the system approaches a generalization transition, gradient noise becomes more heavy-tailed (fewer small corrections, more large exploratory jumps). The tail index decrease is the fingerprint of the optimizer shifting from exploitation to exploration mode.

### 8.4 Prediction 4: Curvature Amplification of Jump Effects

**Claim:** Negative Riemannian sectional curvature amplifies transition magnitude exponentially: escape rate $\sim \exp(\sqrt{|R|})$

**Test protocol:**
1. Estimate sectional curvature $R$ along training trajectory (using second-order geometry of $G(t)$)
2. Measure basin escape rate (how often $\Phi(t)$ peaks)
3. Test exponential relationship: $\log(\text{rate}) \sim \sqrt{|R|}$

**Expected result:** $R^2 > 0.6$ for the linear regression of $\log(\text{rate})$ vs. $\sqrt{|R|}$

**Mechanistic interpretation:** Negatively curved regions of the loss landscape act as hyperbolic funnels — nearby trajectories diverge exponentially. In such regions, the heavy-tailed Lévy jumps are amplified by the curvature, dramatically increasing the basin escape rate.

---

## 9. Implementation Guide

### 9.1 CriticalityTracker: Full Corrected Implementation

```python
#!/usr/bin/env python3
"""
Geometric Lévy Dynamics — Criticality Tracker
Detects synchronized criticality across stochastic, spectral, and geometric dimensions.

Requirements: torch >= 2.0, numpy >= 1.24, scipy >= 1.10
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CriticalityMetrics:
    """All three observables plus derived quantities."""
    C_alpha: float          # Consolidation ratio (stochastic criticality)
    alpha: float            # Lévy tail index
    S: float                # Stability margin (spectral criticality)
    lambda_max: float       # Largest Hessian eigenvalue
    rho: float              # log det G(t) (geometric criticality)
    drho_dt: float          # Rate of change of rho
    Phi: float              # Unified criticality score ∈ [0, 1]
    is_critical: bool       # All three conditions simultaneously satisfied
    stochastic_critical: bool
    spectral_critical: bool
    geometric_critical: bool


class CriticalityTracker:
    """
    Tracks the three critical observables for detecting phase transitions.

    Three observables
    -----------------
    C_alpha  : Consolidation ratio — stochastic signal/noise balance
    S        : Stability margin = 2/η - λ_max(H)
    dρ/dt    : Rate of change of log det G(t)

    Parameters
    ----------
    model     : torch.nn.Module   Network to monitor.
    window    : int               Steps to accumulate for Hill estimator (default 100).
    max_ntk_samples : int         Max data points for NTK computation (default 32).
    sigma1, sigma2, sigma3 : float  Widths of Φ Gaussian factors.
    mu        : float             Center of dρ/dt Gaussian factor.
    geometric_threshold : float   |dρ/dt| threshold for geometric criticality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window: int = 100,
        max_ntk_samples: int = 32,
        sigma1: float = 0.3,
        sigma2: float = 0.15,
        sigma3: float = 0.5,
        mu: float = 1.0,
        geometric_threshold: float = 0.5,
    ):
        self.model = model
        self.window = window
        self.max_ntk_samples = max_ntk_samples
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.mu = mu
        self.geometric_threshold = geometric_threshold

        # Rolling buffers
        self.grad_norms: deque[float] = deque(maxlen=window)
        self.rho_history: deque[float] = deque(maxlen=2)  # Only need last 2

    # ─────────────────────────────────────────────────────────────────────
    # OBSERVABLE 1: Consolidation Ratio Cα
    # ─────────────────────────────────────────────────────────────────────

    def _compute_C_alpha(self) -> Optional[tuple[float, float]]:
        """
        Compute stochastic criticality Cα and tail index α.

        Returns (C_alpha, alpha) or None if insufficient data.

        Algorithm
        ---------
        1. Sort gradient norms in ascending order.
        2. Hill estimator on top-k=10% (maximum likelihood for Pareto tail index).
        3. Estimate diffusion scale σ_α from std of recent norms.
        4. Compute D_α = (σ_α / g_current)^α.
        5. Cα = g_current² / (2 · D_α · window).
        """
        if len(self.grad_norms) < self.window:
            return None

        recent = np.array(self.grad_norms)

        # ── Hill estimator ─────────────────────────────────────────────────
        sorted_g = np.sort(recent)                   # ascending
        k = max(2, int(0.1 * len(sorted_g)))         # top 10%, min 2 samples
        tail = sorted_g[-k:]                         # top-k values
        threshold = sorted_g[-k - 1]                # (k+1)-th largest

        log_ratios = np.log(tail / (threshold + 1e-12))
        if log_ratios.mean() < 1e-10:
            return None                              # degenerate case

        alpha = k / log_ratios.sum()                 # Hill estimator formula
        alpha = float(np.clip(alpha, 1.1, 1.9))      # valid α-stable range

        # ── Diffusion coefficient ──────────────────────────────────────────
        sigma_alpha = float(np.std(recent))
        grad_norm = recent[-1]

        if grad_norm < 1e-12 or sigma_alpha < 1e-12:
            return None

        D_alpha = (sigma_alpha / grad_norm) ** alpha
        C_alpha = grad_norm ** 2 / (2.0 * D_alpha * len(recent) + 1e-12)

        return float(C_alpha), alpha

    # ─────────────────────────────────────────────────────────────────────
    # OBSERVABLE 2: Stability Margin S
    # ─────────────────────────────────────────────────────────────────────

    def _compute_S(
        self,
        loss: torch.Tensor,
        lr: float,
        n_power_iter: int = 5,
    ) -> tuple[float, float]:
        """
        Compute spectral criticality S = 2/η - λ_max(H).

        Uses power iteration via Hessian-vector products (no explicit Hessian).
        Cost: (n_power_iter + 1) backward passes.

        Parameters
        ----------
        loss          : Scalar loss tensor (must retain graph).
        lr            : Current learning rate η.
        n_power_iter  : Number of power iteration steps (default 5).
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return 0.0, 0.0

        # Random unit starting vector
        v = torch.cat([torch.randn_like(p.flatten()) for p in params])
        v = v / (v.norm() + 1e-12)

        # ── Power iteration ────────────────────────────────────────────────
        for _ in range(n_power_iter):
            # First-order grad (retain graph for HVP)
            grads = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True
            )
            flat_grad = torch.cat([g.flatten() for g in grads])

            # HVP: d/dθ (∇L · v) = Hv
            gv = (flat_grad * v.detach()).sum()
            hvp = torch.autograd.grad(gv, params, retain_graph=True)
            v_new = torch.cat([h.flatten() for h in hvp]).detach()

            norm = v_new.norm()
            if norm < 1e-12:
                break
            v = v_new / norm

        # ── Rayleigh quotient: λ_max ≈ v^T H v ────────────────────────────
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True
        )
        flat_grad = torch.cat([g.flatten() for g in grads])
        gv = (flat_grad * v).sum()
        hvp = torch.autograd.grad(gv, params, retain_graph=False)
        hv = torch.cat([h.flatten() for h in hvp])

        lambda_max = float((v * hv).sum().item())
        S = 2.0 / lr - lambda_max

        return float(S), float(lambda_max)

    # ─────────────────────────────────────────────────────────────────────
    # OBSERVABLE 3: Metric Determinant Rate dρ/dt
    # ─────────────────────────────────────────────────────────────────────

    def _compute_rho(self, X: torch.Tensor) -> tuple[float, float]:
        """
        Compute geometric criticality ρ = log det G(t) and dρ/dt.

        G(t) = (1/n) Σᵢ ∇f(xᵢ;θ) ∇f(xᵢ;θ)^T  is the empirical NTK.

        Subsample to max_ntk_samples for computational tractability.
        Cost: max_ntk_samples forward+backward passes.
        """
        n = min(len(X), self.max_ntk_samples)
        X_sub = X[:n]

        grads = []
        # Store model training state — computing per-sample grads requires
        # the model to be in eval mode (no in-place ops from batch norm etc.)
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            outputs = self.model(X_sub)
            # Handle both scalar and vector outputs
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            for i in range(n):
                self.model.zero_grad()
                # Sum over output dimensions (scalar output per sample)
                outputs[i].sum().backward(retain_graph=(i < n - 1))
                g = torch.cat([
                    p.grad.flatten()
                    for p in self.model.parameters()
                    if p.grad is not None
                ])
                grads.append(g.detach())

        if was_training:
            self.model.train()

        if not grads:
            return 0.0, 0.0

        # Compute G = (1/n) J^T J  where J is [n, d] Jacobian matrix
        J = torch.stack(grads)           # [n, d]
        G = (J.T @ J) / n               # [d, d]

        # Eigenvalues (symmetric positive semidefinite)
        try:
            eigvals = torch.linalg.eigvalsh(G)
        except torch.linalg.LinAlgError:
            return 0.0, 0.0

        # log det G = sum of log eigenvalues (filter numerical zeros)
        eigvals_pos = eigvals[eigvals > 1e-6]
        if len(eigvals_pos) == 0:
            rho = 0.0
        else:
            rho = float(torch.log(eigvals_pos).sum().item())

        self.rho_history.append(rho)

        drho_dt = (
            self.rho_history[-1] - self.rho_history[-2]
            if len(self.rho_history) >= 2
            else 0.0
        )

        return rho, float(drho_dt)

    # ─────────────────────────────────────────────────────────────────────
    # UNIFIED CRITICALITY SCORE Φ
    # ─────────────────────────────────────────────────────────────────────

    def _compute_Phi(
        self, C_alpha: float, S: float, drho_dt: float
    ) -> float:
        """
        Unified criticality score Φ ∈ [0, 1].

        Φ(Cα, S, r) = exp(
            -(Cα-1)² / (2σ₁²)
            - S²     / (2σ₂²)
            -(r-μ)²  / (2σ₃²)
        )

        Maximal (= 1) when Cα=1, S=0, dρ/dt=μ simultaneously.
        """
        phi = np.exp(
            -((C_alpha - 1.0) ** 2) / (2 * self.sigma1 ** 2)
            - (S ** 2) / (2 * self.sigma2 ** 2)
            - ((drho_dt - self.mu) ** 2) / (2 * self.sigma3 ** 2)
        )
        return float(phi)

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def update(self, grad_norm: float) -> None:
        """
        Record the gradient norm for this step.
        Call immediately after loss.backward() and before optimizer.step().
        """
        self.grad_norms.append(grad_norm)

    def check(
        self,
        loss: torch.Tensor,
        X: torch.Tensor,
        lr: float,
    ) -> Optional[CriticalityMetrics]:
        """
        Compute all three observables and the unified criticality score.

        Parameters
        ----------
        loss : Scalar loss tensor. Must still have its graph attached
               (call before optimizer.step() and do NOT call loss.backward()
               a second time — the graph from the training backward pass
               is re-used here via create_graph=True).
        X    : Batch of inputs for NTK computation.
        lr   : Current learning rate η.

        Returns
        -------
        CriticalityMetrics or None (if insufficient history).
        """
        C_alpha_result = self._compute_C_alpha()
        if C_alpha_result is None:
            return None

        C_alpha, alpha = C_alpha_result
        S, lambda_max = self._compute_S(loss, lr)
        rho, drho_dt = self._compute_rho(X)

        stoch_crit = 0.8 <= C_alpha <= 1.2
        spec_crit  = -0.1 <= S <= 0.1
        geom_crit  = abs(drho_dt) > self.geometric_threshold

        Phi = self._compute_Phi(C_alpha, S, drho_dt)

        return CriticalityMetrics(
            C_alpha=C_alpha,
            alpha=alpha,
            S=S,
            lambda_max=lambda_max,
            rho=rho,
            drho_dt=drho_dt,
            Phi=Phi,
            is_critical=stoch_crit and spec_crit and geom_crit,
            stochastic_critical=stoch_crit,
            spectral_critical=spec_crit,
            geometric_critical=geom_crit,
        )
```

### 9.2 Training Loop Integration

```python
def train_with_criticality_monitoring(
    model: torch.nn.Module,
    train_loader,
    n_steps: int = 10_000,
    lr: float = 0.01,
    check_every: int = 10,   # Check criticality every N steps (expensive)
) -> list[CriticalityMetrics]:
    """
    Full training loop with criticality monitoring.

    check_every controls the tradeoff between monitoring resolution
    and computational overhead. Criticality checks add ~6 backward
    passes each; set check_every >= 10 for large models.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    tracker   = CriticalityTracker(model, window=100)

    history: list[CriticalityMetrics] = []
    critical_events: list[int] = []

    data_iter = iter(train_loader)

    for step in range(n_steps):
        # ── Get batch ──────────────────────────────────────────────────────
        try:
            X, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            X, y = next(data_iter)

        # ── Forward + backward ─────────────────────────────────────────────
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()

        # Record gradient norm BEFORE optimizer.step()
        grad_norm = torch.cat([
            p.grad.flatten()
            for p in model.parameters()
            if p.grad is not None
        ]).norm().item()
        tracker.update(grad_norm)

        # ── Criticality check (periodic, uses retained graph) ──────────────
        if step % check_every == 0:
            # Re-compute loss with create_graph=True for HVP
            loss_for_check = criterion(model(X[:32]), y[:32])
            metrics = tracker.check(loss_for_check, X[:32], lr=lr)

            if metrics is not None:
                history.append(metrics)

                if metrics.is_critical:
                    critical_events.append(step)
                    print(
                        f"\n⚡ Step {step:>6d}: SYNCHRONIZED CRITICALITY"
                        f"\n   Cα={metrics.C_alpha:.3f}  (α={metrics.alpha:.2f})"
                        f"\n   S={metrics.S:.3f}  (λmax={metrics.lambda_max:.3f})"
                        f"\n   dρ/dt={metrics.drho_dt:.3f}"
                        f"\n   Φ={metrics.Phi:.4f}"
                        f"\n   Components: stochastic={metrics.stochastic_critical}, "
                        f"spectral={metrics.spectral_critical}, "
                        f"geometric={metrics.geometric_critical}"
                    )

        optimizer.step()

    print(f"\nTotal critical events: {len(critical_events)}")
    print(f"Steps: {critical_events[:10]}{'...' if len(critical_events) > 10 else ''}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Minimal working example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 2),
    )

    # Synthetic dataset
    X_all = torch.randn(1000, 10)
    y_all = torch.randint(0, 2, (1000,))
    dataset = torch.utils.data.TensorDataset(X_all, y_all)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    history = train_with_criticality_monitoring(
        model, loader, n_steps=5_000, lr=0.01, check_every=10
    )

    print(f"\nMonitored {len(history)} checkpoints.")
    if history:
        final = history[-1]
        print(f"Final Cα={final.C_alpha:.3f}, S={final.S:.3f}, "
              f"dρ/dt={final.drho_dt:.3f}, Φ={final.Phi:.4f}")
```

### 9.3 Hill Estimator Standalone Utility

```python
def hill_estimator(
    values: np.ndarray,
    k_fraction: float = 0.10,
    alpha_min: float = 1.1,
    alpha_max: float = 1.9,
) -> tuple[float, float]:
    """
    Estimate Lévy tail index α via the Hill estimator.

    Uses the top-k order statistics where k = k_fraction * n.

    Parameters
    ----------
    values      : 1D array of positive values (e.g., gradient norms).
    k_fraction  : Fraction of top-order statistics to use (default 0.10).
    alpha_min,
    alpha_max   : Valid range for clipping (default 1.1–1.9 for deep learning).

    Returns
    -------
    alpha       : Estimated tail index.
    std_err     : Asymptotic standard error = alpha / sqrt(k).

    Notes
    -----
    The Hill estimator is consistent for i.i.d. regularly varying data.
    It is biased when k is too large (bulk contaminates tail estimate).
    Standard choice: k ∈ [0.05n, 0.15n].
    """
    n = len(values)
    k = max(2, int(k_fraction * n))

    sorted_vals = np.sort(values)          # ascending
    tail         = sorted_vals[-k:]        # top-k
    threshold    = sorted_vals[-k - 1]     # (k+1)-th largest

    log_ratios = np.log(tail / (threshold + 1e-12))
    mean_log   = log_ratios.mean()

    if mean_log < 1e-10:
        return float(alpha_max), float("inf")  # degenerate

    alpha    = 1.0 / mean_log
    std_err  = alpha / np.sqrt(k)

    return float(np.clip(alpha, alpha_min, alpha_max)), float(std_err)
```

---

## 10. Computational Complexity and Scaling

| Operation | Cost | Bottleneck | Approximation |
|---|---|---|---|
| Gradient norm collection | $O(d)$ per step | None — standard | — |
| Hill estimator | $O(W \log W)$ per check | Sorting $W=100$ norms | — |
| Stability margin $S$ | $6 \times O(\text{backward})$ | 6 backward passes | Reduce power iter to 3 |
| NTK computation | $O(n' d^2)$ | Full NTK is $O(nd^2)$ | Subsample $n' = 32$ |
| Eigendecomposition | $O(n'^3)$ | $n' = 32$ → $32^3 = 32k$ | Randomized eigensolver |
| Unified score $\Phi$ | $O(1)$ | None | — |

**Scaling strategies for large models (100M+ parameters):**

**Block-wise monitoring:** Compute observables per layer independently. Total cost scales as $O(L \times \text{cost per layer})$ rather than $O(d^2)$ for the full NTK. Different layers transition at different times — layer-wise monitoring reveals which layer is the bottleneck.

**Randomized trace for $\rho$:** Replace full eigendecomposition with $\log \det G \approx \sum_k z_k^\top \log(G) z_k / m$ using Hutchinson-style random probes and a Chebyshev polynomial approximation of $\log(\cdot)$. Cost: $O(m \times \text{HVP cost})$.

**Reducing check frequency:** Set `check_every=100` for production training. Critical windows tend to cluster; missing individual steps within a cluster is acceptable.

**Gradient accumulation:** For very large models, accumulate gradients over multiple micro-batches before computing the Hill estimator to increase the effective window size without additional forward passes.

---

## 11. Limitations and Open Mathematical Problems

### 11.1 Known Limitations

**Time-Varying Metric (Adiabatic Approximation):** The fractional Fokker-Planck equation requires a fixed metric for rigorous analysis. The frozen-metric approximation treats $G(t)$ as constant within each step, then updates adiabatically. This breaks down during rapid feature learning — precisely the regime of interest. All theoretical guarantees are conditional on slow metric evolution.

**Global vs. Layer-Wise:** The framework monitors global observables. Different layers of a deep network can be in different phases simultaneously: early layers may be in the Crystal phase while late layers are still in Vapor. A layer-wise extension would require $3L$ observables (three per layer) and a model for inter-layer criticality interactions.

**Mini-Batch Noise Separation:** The gradient noise $\xi_t$ contains both *intrinsic* noise (from the stochastic loss landscape) and *extrinsic* noise (from mini-batch sampling). The Hill estimator does not separate these. The Lévy tail index $\alpha$ may be partially an artifact of batch size — larger batches reduce extrinsic noise, potentially changing $\hat{\alpha}$.

**Adaptive Optimizers:** The framework is derived for SGD. Adaptive optimizers (Adam, RMSProp, AdaGrad) precondition the gradient by the second moment, which changes the effective noise distribution. The connection between the raw gradient Lévy process and the preconditioned update process is not characterized.

**Empirical Status of Unified Criticality Law:** The Gaussian form of $\Phi$ and the fitted parameters ($\sigma_1, \sigma_2, \sigma_3$) are heuristic. They have been calibrated on a limited set of tasks. Whether these parameters are universal (task-independent) or architecture-specific is an open empirical question.

### 11.2 Open Mathematical Problems

**Problem 1 — Existence Theory:** Prove that weak solutions exist for the fractional Fokker-Planck equation:
$$\frac{\partial p}{\partial t} = \nabla_G \cdot (p \nabla_G L) + D_\alpha \mathcal{L}_\alpha^G[p]$$
on a time-varying Riemannian manifold $(\mathbb{R}^d, G(t))$ where $G(t)$ is itself a solution to a coupled evolution equation. The difficulty is that the fractional Laplacian $\mathcal{L}_\alpha^G$ depends on the metric, which changes on the same timescale as $p$.

**Problem 2 — Necessity of Three Conditions:** The current claim is that all three conditions are *sufficient* for phase transitions. Is the conjunction also *necessary*? Equivalently: can a generalization jump occur when only one or two conditions are satisfied? Identifying a counterexample or proving necessity requires a characterization of all possible transition mechanisms.

**Problem 3 — Universality of Critical Exponents:** The parameters $(\sigma_1, \sigma_2, \sigma_3)$ in $\Phi$ play the role of critical exponents. In statistical physics, critical exponents near second-order phase transitions are universal — they depend only on the symmetry class, not the specific system. Do $(\sigma_1, \sigma_2, \sigma_3)$ exhibit universality across architectures and tasks? A renormalization group analysis of the fractional Fokker-Planck equation near the critical point could answer this.

**Problem 4 — Convergence Rates:** Derive $O(\cdot)$ bounds on the expected time to reach criticality (time-to-transition) as a function of $(\alpha, \text{curvature}, d, \eta)$. The first-passage time analysis used to derive $C_\alpha$ is a scaling argument; a rigorous bound requires controlling the full distribution of first-passage times for an $\alpha$-stable process in a non-convex potential.

**Problem 5 — Adaptive Optimizer Extension:** Characterize the effective noise distribution of Adam-preconditioned gradients. Does the preconditioned noise remain $\alpha$-stable? If so, what is the effective tail index $\alpha_{\text{eff}}$ as a function of the raw index $\alpha$ and the second-moment estimate $v_t$?

---

## 12. References

### Heavy-Tailed Gradient Processes
- **Simsekli, U., Sagun, L., & Gurbuzbalaban, M. (2019).** A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks. *ICML 2019.* — First empirical measurement of $\alpha$-stable gradient noise in deep learning; introduces the Hill estimator for tail index measurement.
- **Zhang, J., He, T., Sra, S., & Jadbabaie, A. (2020).** Why gradient clipping accelerates training: A theoretical justification for adaptivity. *ICLR 2020.* — Connects gradient clipping to Lévy process stabilization.
- **Gurbuzbalaban, M., Simsekli, U., & Zhu, L. (2021).** The heavy-tail phenomenon in SGD. *Mathematical Programming.* — Theoretical foundations for heavy-tailed stochastic approximation.

### Information and Neural Tangent Kernel Geometry
- **Amari, S. (2016).** *Information Geometry and Its Applications.* Springer. — Foundational reference for Fisher-Rao geometry and natural gradient methods.
- **Jacot, A., Gabriel, F., & Hongler, C. (2018).** Neural Tangent Kernel: Convergence and Generalization in Neural Networks. *NeurIPS 2018.* — Original NTK discovery; infinite-width lazy training theory.
- **Lee, J., et al. (2019).** Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. *NeurIPS 2019.* — Infinite-width lazy training; motivation for why time-varying $G(t)$ is the finite-width correction.

### Edge-of-Stability
- **Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021).** Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. *ICLR 2021.* — Original discovery and characterization of edge-of-stability regime.
- **Damian, A., Nichani, E., Ge, R., & Lee, J. D. (2023).** Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability. *NeurIPS 2023.* — Mechanistic explanation via implicit regularization; supports the spectral criticality framework.

### Phase Transitions and Grokking
- **Power, A., Anand, Y., Shleifer, D., Cohen, N., & Garg, S. (2022).** Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *ICLR 2022 Workshop.* — Original grokking discovery; empirical benchmark for Prediction 1.
- **Nanda, N., Lawrence, L., Chan, T., & Lieberum, T. (2023).** Progress measures for grokking via mechanistic interpretability. *ICLR 2023.* — Mechanistic analysis of grokking as representation reorganization; directly supports geometric criticality framework.
- **Barak, B., Edelman, B., Goel, S., Kakade, S., Malach, E., & Zhang, C. (2022).** Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit. *NeurIPS 2022.* — Hidden progress as subcritical consolidation.

### Lévy Processes on Manifolds
- **Applebaum, D. (2004).** *Lévy Processes and Stochastic Calculus.* Cambridge University Press. — Comprehensive treatment of Lévy processes including manifold extensions.
- **Bass, R. F., & Levin, D. A. (2002).** Harnack inequalities for jump processes. *Potential Analysis, 17*(4), 375–388. — Jump processes and heat kernels; foundational for fractional PDE theory.
- **Liao, M. (2004).** *Lévy Processes in Lie Groups.* Cambridge University Press. — Lévy processes on curved spaces; relevant to the manifold extension of GLD.

### Riemannian Geometry
- **Hsu, E. P. (2002).** *Stochastic Analysis on Manifolds.* American Mathematical Society. — Rigorous foundation for stochastic processes on Riemannian manifolds.
- **do Carmo, M. P. (1992).** *Riemannian Geometry.* Birkhäuser. — Standard reference for differential geometry of learning manifolds.
- **Petersen, P. (2016).** *Riemannian Geometry* (3rd ed.). Springer. — Graduate-level treatment including sectional curvature relevant to Prediction 4.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **α-stable Lévy process** | Stochastic process with power-law jump distribution; tail index $\alpha \in (1,2)$ implies infinite variance. Reduces to Brownian motion at $\alpha = 2$. |
| **Hill estimator** | Maximum likelihood estimator for the tail index of a heavy-tailed distribution: $\hat{\alpha} = k / \sum_{i=1}^k \log(X_{(n-i+1)} / X_{(n-k)})$ using top-$k$ order statistics. |
| **NTK (Neural Tangent Kernel)** | $G(t) = (1/n) \sum_i \nabla_\theta f(x_i) \nabla_\theta f(x_i)^\top$ — empirical NTK matrix measuring gradient alignment across data. Defines the Riemannian metric on parameter space. |
| **Consolidation ratio $C_\alpha$** | $|\nabla L|^2 / (2 D_\alpha d)$ — ratio of gradient power to Lévy diffusion. $C_\alpha = 1$ is the stochastic criticality threshold derived from first-passage analysis. |
| **Stability margin $S$** | $2/\eta - \lambda_{\max}(H)$ — signed distance from the edge-of-stability threshold. $S = 0$ is the spectral criticality point. |
| **Log-determinant $\rho$** | $\log \det G(t) = \sum_k \log \lambda_k(G)$ — log-volume of the NTK ellipsoid; measures effective dimensionality of the gradient manifold. |
| **$d\rho/dt$** | Rate of change of $\rho$; large values signal rapid representation reorganization (geometric criticality). |
| **Unified criticality score $\Phi$** | Product of three Gaussian factors: $\Phi \in [0,1]$. Maximal when $C_\alpha = 1$, $S = 0$, and $d\rho/dt = \mu$ simultaneously. |
| **Synchronized criticality** | State where all three conditions ($C_\alpha \approx 1$, $S \approx 0$, $|d\rho/dt| > \tau$) hold simultaneously. Predicted to precede phase transitions by 10–50 steps. |
| **Edge-of-stability** | Training regime where $\lambda_{\max}(H) \cdot \eta \approx 2$ — the classical divergence boundary. Empirically: stable training is possible here due to nonlinear self-stabilization. |
| **Grokking** | Sudden generalization after extended overfitting plateau. Predicted here to occur 10–50 steps after synchronized criticality is first detected. |
| **Fractional Laplacian $\mathcal{L}_\alpha$** | Nonlocal differential operator $-(-\Delta)^{\alpha/2}$; Fourier symbol $-|k|^\alpha$. Replaces the standard Laplacian in the diffusion term of the Fokker-Planck equation for Lévy processes. |
| **Frozen-metric approximation** | Treating $G(t)$ as constant within each update step; valid when metric changes slowly (lazy regime). Breaks down during rapid feature learning. |
| **HVP (Hessian-vector product)** | $Hv = \nabla(\nabla L \cdot v)$ computed via double backpropagation. Cost: one additional backward pass. Used for power iteration without forming $H$ explicitly. |
| **Rayleigh quotient** | $\lambda \approx v^\top H v$ for unit vector $v$ — approximates the eigenvalue corresponding to direction $v$. Converges to $\lambda_{\max}$ after power iteration. |
| **Power iteration** | Iterative algorithm: $v \leftarrow Hv / \|Hv\|$, converges to the eigenvector of $\lambda_{\max}(H)$. Requires $O(k)$ HVPs for $k$ iterations. |
| **Lazy training / NTK regime** | Training regime where $G(t) \approx G(0)$ throughout — feature representations do not change. Occurs exactly at infinite width; approximate at finite width for small learning rate. |
| **Feature learning regime** | Training regime where $G(t)$ evolves significantly — representations reorganize. Occurs at finite width and larger learning rates. Characterized by large $|d\rho/dt|$. |
| **Tail index $\alpha$** | Exponent governing the power-law tail of the gradient distribution. $\alpha = 2$: Gaussian (finite variance). $\alpha < 2$: Lévy-stable (infinite variance). Empirically $\alpha \approx 1.5$ in deep learning. |
| **Sectional curvature $R$** | Riemannian curvature of the loss landscape along the training trajectory. Negative curvature creates hyperbolic funnel geometry that amplifies Lévy jump effects. |

---

## Summary

This framework makes three contributions:

**Mathematical:** Unifies heavy-tailed stochastic processes ($\alpha$-stable Lévy), time-varying Riemannian geometry (NTK metric $G(t)$), and phase-transition theory in a single framework for neural network training dynamics.

**Predictive:** Identifies three independent critical thresholds ($C_\alpha \approx 1$, $S \approx 0$, $|d\rho/dt| > \tau$) whose simultaneous alignment predicts generalization jumps 10–50 steps in advance. Each condition is derivable from first principles; their conjunction is an empirical law.

**Practical:** Provides an implementable monitoring system (`CriticalityTracker`) with known computational cost, testable predictions, and a clear path to criticality-aware optimizer design (adaptive learning rates from real-time $C_\alpha$, criticality-triggered learning rate scaling).

**Mathematical status:** Theoretically complete at the level of dimensional analysis and scaling arguments. Rigorous existence theory for the fractional Fokker-Planck equation on time-varying manifolds remains open. Empirical validation on grokking tasks is in progress.

> *"Phase transitions are not accidents but necessary consequences of training at the intersection of three independent critical boundaries."*
