# 02460 Advanced Machine Learning — Exam Cheat Sheet
*(3 pages: Page 1 = Module 1 VAE+Flows, Page 2 = Diffusion+Riemannian, Page 3 = Graphs)*

---

# PAGE 1 — MODULE 1: Deep Generative Models

## KL Divergence
```
KL(p‖q) = ∫p(x) log[p(x)/q(x)] dx = E_{x~p}[log p(x) - log q(x)]
```
- **Properties**: KL ≥ 0, KL = 0 ⟺ p = q, asymmetric
- **KL between univariate Gaussians** (CLOSED FORM — exam favourite):
```
KL[N(μ₁,σ₁) ‖ N(μ₂,σ₂)] = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```
- **KL between multivariate Gaussians** N₀(μ₀,Σ₀) vs N₁(μ₁,Σ₁):
```
KL(N₀‖N₁) = ½[log det(Σ₁Σ₀⁻¹) + (μ₀-μ₁)ᵀΣ₁⁻¹(μ₀-μ₁) + tr(Σ₁⁻¹Σ₀) - D]
```
- **KL(N(μ,σ²) ‖ N(0,1))** = ½(σ² + μ² - 1 - log σ²) [special case, diagonal]

## MLE = KL minimization
```
θ̂ = argmax_θ Σᵢ log p_θ(xᵢ)   ⟺   argmin_θ KL(p_data ‖ p_θ)
```
(The -E[log p_data(x)] term doesn't depend on θ)

## Jensen's Inequality
For concave g: g(E[X]) ≥ E[g(X)]  → used to derive ELBO

## DLVM & VAE

**Model:**  z ~ p(z),   x ~ p_θ(x|z)  
**Marginal** (intractable for nonlinear decoder):  p(x) = ∫ p_θ(x|z) p(z) dz

**ELBO derivation** (introduce q_φ(z|x) > 0):
```
log p(x) = log E_{z~q_φ}[p_θ(x|z)p(z)/q_φ(z|x)]
          ≥ E_{z~q_φ}[log p_θ(x|z)p(z)/q_φ(z|x)]   [Jensen's]
          = E_{z~q_φ}[log p_θ(x|z)] - KL(q_φ(z|x) ‖ p(z))
          =: L(θ,φ)    [ELBO]
```
**Equivalently:**  log p(x) = L(θ,φ) + KL(q_φ(z|x) ‖ p_θ(z|x))  
→ Maximising ELBO wrt (θ,φ): maximise recon, minimise KL to posterior

**ELBO decompositions:**
```
L = E[log p_θ(x|z)] - KL(q_φ(z|x) ‖ p(z))          [reconstruction - regulariser]
  = log p(x) - KL(q_φ(z|x) ‖ p_θ(z|x))              [tight iff q = true posterior]
```
**Amortized VI:** q_φ(z|x) = Ψ(z | g_φ(x))  where g_φ is the encoder NN  
**Standard choice:** q_φ(z|x) = N(z | μ_φ(x), diag σ_φ²(x))

**Reparameterization trick:**
```
z = μ_φ(x) + σ_φ(x) ⊙ ε,   ε ~ N(0, I)
∇_φ L = E_{ε~N(0,I)}[∇_φ log p_θ(x | μ_φ(x) + σ_φ(x)⊙ε)] - ∇_φ KL(q_φ‖p)
```
- Moves gradient through sampling; low variance estimator
- Works for Gaussian, Beta, Gamma; NOT for discrete (use Gumbel-softmax)

**KL for diagonal Gaussians vs N(0,I) prior** (closed form used in VAE):
```
KL(N(μ,diag σ²) ‖ N(0,I)) = ½ Σⱼ [σⱼ² + μⱼ² - 1 - log σⱼ²]
```

**VAE issues:**
- **Posterior collapse**: q_φ(z|x) → p(z), decoder ignores z → use KL warm-up
- **Hole problem**: aggregated posterior q_φ(z) = (1/N)Σ q_φ(z|xₙ) ≠ p(z) → unrealistic samples from holes

**Improving the prior:**
```
Aggregated posterior: q_φ(z) = (1/N)Σₙ q_φ(z|xₙ)   [best prior, minimises cross-entropy]
MoG prior: p_λ(z) = Σₖ wₖ N(z|μₖ, diag σₖ²)        [K < N learnable]
VampPrior: p_λ(z) = (1/K)Σₖ q_φ(z|uₖ)              [pseudo-inputs uₖ learnable]
```

**Hierarchical VAE** (2-level), bottom-up: Q(z₁,z₂|x) = q(z₂|z₁)q(z₁|x)
```
ELBO = E[log p(x|z₁)] - E[log q(z₁|x)/p(z₁|z₂)] - E[KL(q(z₂|z₁)‖p(z₂))]
```

## Normalizing Flows

**Model:** u ~ p_φ(u),  x = T(u), where T is a diffeomorphism (bijective, C¹ with C¹ inverse)

**Change of variables:**
```
p_x(x) = p_u(u) |det J_T(u)|⁻¹  where u = T⁻¹(x)
       = p_u(T⁻¹(x)) |det J_{T⁻¹}(x)|      [inverse function theorem: J_{T⁻¹} = J_T⁻¹]
```

**Log-likelihood for MLE:**
```
ℓ(ψ,φ) = Σᵢ [log p_φ(T⁻¹(xᵢ)) + log|det J_{T⁻¹}(xᵢ)|]
```
Only need T⁻¹ and J_{T⁻¹} for MLE; need T for sampling.

**Composition:** T = T_K ∘ ⋯ ∘ T₁
```
log|det J_T| = Σₖ log|det J_{Tₖ}|    [chain rule for Jacobians]
```

**Affine coupling layer (RealNVP):** partition z = (z_{1:d}, z_{d+1:D})
```
Forward:   z'_{1:d} = z_{1:d}
           z'_{d+1:D} = exp(s(z_{1:d})) ⊙ z_{d+1:D} + t(z_{1:d})
Inverse:   z_{1:d} = z'_{1:d}
           z_{d+1:D} = (z'_{d+1:D} - t(z_{1:d})) ⊙ exp(-s(z_{1:d}))
```
where s,t : ℝᵈ → ℝᴰ⁻ᵈ are arbitrary neural nets.

**Jacobian** (triangular → easy determinant):
```
J_T = [I_{d×d}      0      ]
      [∂z'_{d+1:D}/∂z_{1:d}   diag(exp(s(z_{1:d})))]

det J_T = ∏ᵢ exp(s(z_{1:d})ᵢ) = exp(Σᵢ s(z_{1:d})ᵢ)
```
**Permutation layer:** det J = ±1 (volume preserving), needed so all dims get transformed.

**Flow as VAE prior** — ELBO with flow prior p_λ(z) = p_φ(T⁻¹(z))|det J_T(T⁻¹(z))|⁻¹:
```
L = E_{z~q_φ}[log p_θ(x|z)] + E_{z~q_φ}[log p_φ(T⁻¹(z)) - log|det J_T(T⁻¹(z))|] - E[log q_φ(z|x)]
  = recon + [extra log-det term] - KL-like term
```

---

# PAGE 2 — DIFFUSION MODELS & RIEMANNIAN GEOMETRY

## DDPM (Denoising Diffusion Probabilistic Models)

**Forward process** (Markov chain, fixed, no parameters):
```
q(zₜ|z_{t-1}) = N(zₜ | √(1-βₜ) z_{t-1}, βₜ I)
```
**Marginal** (reparameterize, skip all intermediate steps):
```
q(zₜ|x) = N(zₜ | √ᾱₜ x, (1-ᾱₜ)I),   where ᾱₜ = ∏ₛ₌₁ᵗ (1-βₛ)
```
**Key**: ᾱ_T → 0 as T→∞ (show: ᾱ_T < (1-β₁)^T → 0) ⟹ q(z_T|x) → N(0,I)

**Posterior** (Gaussian, closed form!):
```
q(z_{t-1}|zₜ,x) = N(z_{t-1} | μ̃ₜ(zₜ,x), β̃ₜ I)
μ̃ₜ(zₜ,x) = [√ᾱ_{t-1} βₜ/(1-ᾱₜ)] x + [√αₜ(1-ᾱ_{t-1})/(1-ᾱₜ)] zₜ
β̃ₜ = (1-ᾱ_{t-1})/(1-ᾱₜ) · βₜ,   where αₜ = 1-βₜ
```

**Reverse process** (learned, Gaussian):
```
p_θ(z_{t-1}|zₜ) = N(z_{t-1} | μ_θ(zₜ,t), σₜ²I),   p_θ(z_T) = N(0,I)
```

**ELBO** (hierarchical VAE interpretation):
```
log p(x) ≥ L = -KL(q(z_T|x)‖p(z_T)) - Σ_{t≥2} E[KL(q(z_{t-1}|zₜ,x) ‖ p_θ(z_{t-1}|zₜ))] + E[log p(x|z₁)]
              =: L_T + Σ L_{t-1} + L_0
```
**L_{t-1} simplified** (using KL of two Gaussians):
```
L_{t-1} = (1/2σₜ²) ‖μ̃ₜ(zₜ,x) - μ_θ(zₜ,t)‖² + C   [C doesn't depend on θ]
```

**Predict noise** (Ho et al. 2020): reparametrize zₜ = √ᾱₜ x + √(1-ᾱₜ) ε, ε~N(0,I)
```
μ̃ₜ(zₜ,x) can be written as: (1/√αₜ)[zₜ - (βₜ/√(1-ᾱₜ)) ε]
⟹ train ε_θ(zₜ,t) to predict ε:
L_{t-1} ≈ ‖ε - ε_θ(√ᾱₜ x + √(1-ᾱₜ)ε, t)‖²   [simplified, drop pre-factor]
```

**Training algorithm:**
```
1. t ~ U{1,...,T};  ε ~ N(0,I);  x ~ p_data
2. zₜ = √ᾱₜ x + √(1-ᾱₜ) ε
3. L = ‖ε - ε_θ(zₜ, t)‖²;  gradient step
```

**Sampling (ancestral):**
```
z_T ~ N(0,I)
for t=T,...,1:  η ~ N(0,I)
   z_{t-1} = (1/√αₜ)[zₜ - (βₜ/√(1-ᾱₜ)) ε_θ(zₜ,t)] + √βₜ η
```

## SDE-based Diffusion (VP-SDE ≈ DDPM continuous limit)

**Forward SDE:**  dx = -½β(t)x dt + √β(t) dw  [drift + diffusion]

**Reverse SDE** (requires score ∇_{xₜ} log p(xₜ)):
```
dx = [f(x,t) - g(t)² ∇_{xₜ} log p(xₜ)] dt + g(t) dw̃
```
**Probability flow ODE** (same marginals, deterministic):
```
dx/dt = f(x,t) - ½g(t)² ∇_{xₜ} log p(xₜ)
```
**Learn score**: s_θ(xₜ,t) ≈ ∇_{xₜ} log p(xₜ), via denoising score matching (DSM):
```
L_DSM = E_{t,x₀,xₜ}[λ(t) ‖s_θ(xₜ,t) - ∇_{xₜ} log p(xₜ|x₀)‖²]
```
If forward SDE has linear drift: p(xₜ|x₀) is Gaussian → tractable!  
Since ∇_{xₜ} log N(xₜ|μ,σ²I) = (μ-xₜ)/σ² = -ε/σ → score ∝ -ε/σ

## FID (Fréchet Inception Distance)
```
FID = ‖μ_r - μ_g‖² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
```
Embed images via Inception-v3, fit Gaussians to real/generated features, compute Fréchet distance. Lower = better.

## Module 2: Riemannian Geometry & Identifiability

**Identifiability:** model p_θ identifiable iff θ ↦ p_θ is injective (bijective). VAEs/NNs are NOT identifiable (can reparametrize latent space without changing model fit).

**Manifold:** image M = f(Z) where f : Z ⊆ ℝᴹ → ℝᴰ smooth (D > M)  
**Immersed** iff J_f full rank for all z (immersed ⊃ embedded)  
MLP with full-rank weight matrices + smooth activations → immersed manifold.

**Pullback Riemannian metric** (from Euclidean observation space):
```
M(z) = J_f(z)ᵀ J_f(z)   [M×M positive semidefinite matrix]
```
Inner product in tangent space: ⟨u,v⟩_z = uᵀM(z)v  
**Identifiable**: distances/angles in z computed via M are invariant to reparametrization.

**Curve length on manifold:**
```
L(c) = ∫₀¹ √(ċ(t)ᵀ M(c(t)) ċ(t)) dt   [c: [0,1]→Z latent curve]
```
**Curve energy** (Cauchy-Schwarz gives E(c) ≥ L(c)²):
```
E(c) = ∫₀¹ ċ(t)ᵀ M(c(t)) ċ(t) dt
```
Equality iff constant speed (‖ċ‖_M = const). Energy minimizers = geodesics.

**Geodesic** = shortest path = energy minimizer (constant speed):
```
Geodesic ODE: c̈ᵢ + Σⱼₖ Γⁱⱼₖ ċⱼ ċₖ = 0
```
Two BVP types: boundary value (start+end) or initial value (start+velocity → unique by Picard-Lindelöf).

**Log map** (Riemannian subtraction): Log_p(q) = initial velocity of geodesic p→q (tangent vector)  
**Exp map** (Riemannian addition): Exp_p(v) = endpoint of geodesic from p with velocity v

**Noisy manifold (VAE with Gaussian output):** p_θ(x|z) = N(x | μ_θ(z), diag σ_θ²(z))
```
Rewrite: x = μ_θ(z) + Λ_θ(z)^{1/2} η,  η~N(0,I)   [random projection]
Metric: M(z) = J_μᵀ J_μ + E[J_Λ^{1/2}ᵀ J_Λ^{1/2}] ≈ J_μᵀ J_μ + term from uncertainty
```
Uncertainty term creates "walls" around data → geodesics follow data trends.  
**Ensemble**: M models → uncertainty from variance of predictions.

**Fisher-Rao metric** (information geometry): KL divergence ≈ local quadratic form:
```
KL(p(x;θ) ‖ p(x;θ+dθ)) ≈ ½ dθᵀ F(θ) dθ
F(θ)ᵢⱼ = E_{x~p(x;θ)}[∂ᵢ log p · ∂ⱼ log p]   [Fisher information matrix]
```
Parameter manifold of NN: metric G(θ) = E_x[J_f(x;θ)ᵀ J_f(x;θ)]  
Kernel manifold = zero-length curves (same function, different params) → Gaussian samples in param space → project to kernel for UQ.

---

# PAGE 3 — MODULE 3: Graph Models

## Graph Definitions

**Simple graph:** G = (V,E), adjacency A ∈ {0,1}^{N×N}, degree D = diag(d₁,...,d_N)

**Laplacians:**
```
L = D - A                              [unnormalized]
L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}   [symmetric normalized]
L_rw = D⁻¹L = I - D⁻¹A               [random walk]
```
All have non-negative eigenvalues.

## Graph Statistics

**Eigenvector centrality:** λe = Ae  (solve for largest eigenvalue eigenvector)
```
eᵤ = (1/λ) Σ_{v∈N(u)} eᵥ   [recursive: score = scaled sum of neighbors' scores]
```
Perron-Frobenius: largest eigenvalue unique, eigenvector non-negative.

**Clustering coefficient:**
```
cᵤ = |{(v₁,v₂)∈E: v₁,v₂∈N(u)}| / C(dᵤ,2) = #{triangles at u} / #{possible triangles at u}
c = (D(D-I))⁻¹ diag(A³)   [matrix form; not computable by simple message passing]
```
Example: cᵤ = A³ᵤᵤ / (dᵤ(dᵤ-1))

**Weisfeiler-Lehman (WL) test** (graph isomorphism):
```
1. Init: l_v^{(0)} = d_v   (degree)
2. Iterate: l_v^{(i)} = hash({{l_u^{(i-1)} : u ∈ N(v)}})   (multi-set hash)
3. Summarize: hash({{l_v^{(i)} : v ∈ V}})
4. If summaries differ → not isomorphic. WL cannot distinguish all non-isomorphic graphs.
```

## Message Passing

**General form:**
```
h_u^{(k+1)} = update^{(k)}(h_u^{(k)}, aggregate^{(k)}({h_v^{(k)}: v∈N(u)}))
y = readout({h₁,...,h_N})
```
**Basic GNN:**
```
h_u^{(k)} = σ(W_self h_u^{(k-1)} + W_neigh Σ_{v∈N(u)} h_v^{(k-1)} + b)
```
**Normalized aggregation** (handle degree variability):
```
m_{N(u)} = Σ_{v∈N(u)} h_v / |N(u)|   or   Σ_{v∈N(u)} h_v / √(|N(u)||N(v)|)
```
**Set pooling** (universal): m_{N(u)} = MLP_θ(Σ_{v∈N(u)} MLP_φ(h_v))

**Attention:** αᵤ,ᵥ = softmax_{v∈N(u)}(hᵤᵀWh_v),  m_{N(u)} = Σ αᵤ,ᵥ hᵥ  
**Scaled dot-product attention:** αᵤ,ᵥ = softmax(kᵤᵀqᵥ/√d), m = Σ αᵤ,ᵥ vᵥ

**Skip connections / gating** (reduce over-smoothing):
```
Residual: h^new = h + update_base(h, m_{N(u)})
GRU:  r = σ(Wmr m + Whr h + br)            [reset gate]
      z = σ(Wmz m + Whz h + bz)            [update gate]
      h̄ = tanh(Wmh m + Whh(r⊙h) + bh)    [candidate]
      h^new = z⊙h̄ + (1-z)⊙h
```

## Node Embeddings (Shallow)

**Encoder:** enc(u) = Zᵀsᵤ (lookup table; sᵤ one-hot), Z ∈ ℝ^{N×d}

**Decoders:**
```
Dot product:   dec(zᵤ,zᵥ) = zᵤᵀzᵥ         → ℝ
Squared dist:  dec(zᵤ,zᵥ) = ‖zᵤ-zᵥ‖²     → ℝ₊
Sigmoid:       dec(zᵤ,zᵥ) = σ(zᵤᵀzᵥ+b)   → [0,1]
Softmax:       dec(zᵤ,zᵥ) = exp(zᵤᵀzᵥ)/Σ_w exp(zᵤᵀzw)
```

**Loss functions:**
```
MSE:   L_mse = Σ_{(u,v)∈D} (Sᵤᵥ - zᵤᵀzᵥ)² = ‖S - ZZᵀ‖_F²

BCE:   L_bce = Σ_{(u,v)∈D} [-Sᵤᵥ log σ(zᵤᵀzᵥ+b) - (1-Sᵤᵥ) log(1-σ(zᵤᵀzᵥ+b))]
             = Σ_{(u,v)∈E} -log σ(zᵤᵀzᵥ+b) + Σ_{(u,v)∉E} -log σ(-zᵤᵀzᵥ-b)

Random walk:  L_rw = Σ_{(u,v)∈W} -log[exp(zᵤᵀzᵥ) / Σ_w exp(zᵤᵀzw)]
              ≈ Σ_{(u,v)∈W} [-log σ(zᵤᵀzᵥ+b) - γ E_{w~P(w)}[log σ(-zᵤᵀzw-b)]]  [negative sampling]
```
Note: 1-σ(x) = σ(-x). Shallow embeddings cannot do induction (unseen nodes).

## Graph Convolutions & Fourier Transform

**Graph convolution** (using A as shift operator):
```
y = (h[0]I + h[1]A + h[2]A² + ⋯) x = Σₖ h[k] Aᵏ x
```

**Discrete Fourier transform** (DFT):
```
x̃[k] = Σₙ x[n] ωₙᵏⁿ,   ωₙ = e^{-i2π/N}   (UH matrix: UHₖₙ = ωₙᵏⁿ/√N)
DFT: x̃ = UHx,   inverse: x = Ux   (U unitary: UHU = I)
Convolution theorem: y = Hx ⟺ ỹ[k] = x̃[k]·h̃[k]
```
**Proof:** ỹ[k] = Σₙ(Σₗ x[ℓ]h[(n-ℓ)mod N])ωₙᵏⁿ = x̃[k]·h̃[k]

**Graph Fourier transform:**  
Eigendecompose adjacency: A = UΛUᴴ (eigenvectors = graph frequency basis)  
For cycle graph: eigenvectors = DFT basis vectors (you should be able to show this).

**Graph convolution via Fourier:**
```
y = U(Σₖ h[k]Λᵏ)UH x   [equivalent to spatial form; compute in frequency domain]
```

## GNNs and Probabilistic Models

**Markov Random Field** (graph as graphical model):
```
p({xᵥ},{zᵥ}) ∝ Πᵥ Φ(xᵥ,zᵥ) · Π_{(u,v)∈E} Ψ(zᵤ,zᵥ)
```
**Mean field variational inference** — factored posterior q = Πᵥ qᵥ(zᵥ), minimise KL:
```
Fixed point: log q_v^{(t+1)}(zᵥ) = cᵥ + log Φ(xᵥ,zᵥ) + Σ_{u∈N(v)} ∫ qᵤ(zᵤ) log Ψ(zᵤ,zᵥ) dzᵤ
```
This is message passing! Kernel mean embedding: µᵥ = ∫ ϕ(z)qᵥ(z)dz embedded in Hilbert space → GNN updates.

## Graph VAE

**Node-level VAE** (encoder = GNN, decoder = pairwise):
```
q_φ(Z|G): μ_Z, log σ_Z = GNN(A,X)   [outputs per node]
Z = μ_Z + ε⊙σ_Z,  ε~N(0,I)         [reparameterization]
p_θ(Aᵤᵥ=1|Z) = σ(zᵤᵀzᵥ + b)        [Bernoulli decoder]
Prior: p(Z) = N(0,I)
ELBO: L = E_{q_φ}[Σ_{u,v} log p_θ(Aᵤᵥ|Z)] - KL[q_φ(Z|G) ‖ p(Z)]
```

**Graph-level VAE** (whole graph → single z):
```
μ_z, log σ_z = GNN(A,X) [readout]
z = μ_z + ε⊙σ_z
Ã = σ(MLP(z))   [N×N edge probabilities]
Issues: fixed node count; node ordering ambiguity
```

**GANs for graphs:** Generator: Ã = σ(MLP(z));  Discriminator: GNN classifier (permutation invariant)
```
min_θ max_φ E_{x~p(x)}[log(1-d_φ(x))] + E_{z~p(z)}[log d_φ(g_θ(z))]
```

## Symmetry in Geometric GNNs

**Invariant:** f(T(h)) = f(h) (output unchanged under transformation)  
**Equivariant:** f(T(h)) = T'(f(h)) (output transforms with input)

Invariant operations: ‖v‖, v₁·v₂, ‖v₁×v₂‖  
Equivariant operations: a·v, v₁+v₂, v₁×v₂

## Evaluation of Graph Generation

**Graph statistics to compare:** degree distribution, clustering coefficient, eigenvector centrality

**Total variation distance:**
```
d(s_gen, s_test) = sup_x |P(s_gen ∈ x) - P(s_test ∈ x)|
```
For discrete distributions: d = ½ L₁ distance between histograms.

**Traditional generative models:**
- **Erdős-Rényi:** P(Aᵤᵥ=1) = r (all edges iid)
- **Stochastic block model:** nodes assigned to blocks; P(Aᵤᵥ=1) = rᵢⱼ (depends on blocks)
- **Preferential attachment:** P(Aᵤᵥ=1) ∝ d_v^{(t)} (rich get richer)

---

## Key model comparison table

| Model     | Training  | Likelihood | Invertible | Bottleneck | Sampling |
|-----------|-----------|------------|------------|------------|---------|
| VAE       | Stable    | Approx     | No         | Yes        | Fast    |
| Flow      | Stable    | Exact      | Yes        | No         | Fast    |
| DDPM      | Stable    | Approx     | No         | No         | Slow    |
| SDE/SBGM  | Stable    | ~Exact(ODE)| No/Yes     | No         | Slow    |

## Quick Identities

- **PPCA MLE bias**: b̂ = x̄ (mean of data) — show by ∂ℓ/∂b = 0 → C⁻¹(b-x̄) = 0
- **Jensen's**: for concave g: g(E[X]) ≥ E[g(X)] → log E[X] ≥ E[log X]
- **Cauchy-Schwarz**: (∫f·g)² ≤ (∫f²)(∫g²); equality iff f∝g
- **Energy bound**: L(c)² ≤ E(c) with equality iff constant speed
- **Gradient of Gaussian log-score**: ∇_x log N(x|μ,σ²I) = (μ-x)/σ² = -ε/σ (where x=μ+σε)
- **Matrix det (triangular)**: det A = ∏ᵢ Aᵢᵢ
- **Chain rule Jacobian**: J_{T∘S}(x) = J_T(S(x)) · J_S(x)
- **Aggregate posterior**: q_φ(z) = (1/N)Σₙ q_φ(z|xₙ) [aggregated posterior]
