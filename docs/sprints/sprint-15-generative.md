# Sprint 15 — Generative Track

**Theme:** VAE, GAN, DCGAN, DDPM diffusion models
**Duration:** 4 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Implement the three major generative model families from scratch: variational autoencoders, generative adversarial networks, and denoising diffusion models. By end of sprint you ship a generation API that produces images from class labels.

---

## Concepts

### Variational Autoencoder (VAE)

- Encoder: `q(z|x)` — maps input to distribution parameters `(μ, log σ²)`.
- Decoder: `p(x|z)` — maps latent sample back to reconstruction.
- Reparameterization trick: `z = μ + σ * ε` where `ε ~ N(0,1)` — makes sampling differentiable.
- Loss: `L = reconstruction_loss + KL(q(z|x) || N(0,1))`.
- KL term: `-0.5 * sum(1 + log σ² - μ² - σ²)`.

### GAN Training Dynamics

- **Generator G:** maps noise `z ~ N(0,1)` to fake image.
- **Discriminator D:** classifies images as real (1) or fake (0).
- Minimax loss: `min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]`.
- In practice: train D for k steps, then train G for 1 step.
- Mode collapse: G learns to produce one mode of the data — a fundamental training instability.

### DCGAN

- Replace Linear layers with Conv and ConvTranspose.
- Generator: noise → ConvTranspose2d blocks → image (upsample from 4×4 to 64×64).
- Discriminator: image → Conv2d blocks → scalar (downsample from 64×64 to 1×1).
- BatchNorm in both G and D (except last layer of G and first layer of D).

### Denoising Diffusion Probabilistic Models (DDPM)

- **Forward process:** gradually add Gaussian noise over T=1000 steps: `q(x_t|x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)`.
- **Noise schedule:** `β_t` increases from β_1=0.0001 to β_T=0.02 (linear).
- **Closed form:** `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` where `ᾱ_t = prod(1-β_i)` and `ε ~ N(0,1)`.
- **Reverse process:** learn `p_θ(x_{t-1}|x_t)` — a UNet that predicts the noise `ε` added at step t.
- **Training:** sample t uniformly, add noise to x_0, predict ε with UNet, minimize MSE.
- **Sampling:** start from `x_T ~ N(0,1)`, iteratively denoise using predicted noise.

### UNet for Diffusion

- Encoder (downsample) + bottleneck + decoder (upsample) with skip connections.
- Time step embedding: `t` is embedded via sinusoidal encoding + MLP, added to each ResBlock.
- ResBlock: `Conv-GroupNorm-SiLU-Conv + time_emb_proj + skip`.

---

## Tasks

- [ ] Implement VAE encoder: `Conv → Flatten → Linear → (μ, log_var)`
- [ ] Implement reparameterization trick: `z = μ + exp(0.5*log_var) * randn`
- [ ] Implement VAE decoder: `Linear → Reshape → ConvTranspose → Sigmoid`
- [ ] Implement VAE loss: reconstruction (BCE or MSE) + KL divergence term
- [ ] Train VAE on CelebA 64×64 — sample from prior, interpolate between two latent codes
- [ ] Implement GAN generator and discriminator for 64×64 images
- [ ] Implement DCGAN with all BatchNorm conventions — train on CelebA
- [ ] Implement FID score computation (Fréchet Inception Distance) for evaluating generation quality
- [ ] Implement DDPM noise schedule: β_t linear, compute ᾱ_t cumulative products
- [ ] Implement UNet denoiser with time embedding, ResBlocks, GroupNorm, skip connections
- [ ] Implement DDPM forward process: add noise at arbitrary step t in closed form
- [ ] Train DDPM on CIFAR-10 — 200 epochs minimum — visualize denoising steps
- [ ] Implement DDPM reverse sampling: iterative denoising from pure noise
- [ ] Read DDPM paper — annotate in `papers/ddpm.md`
- [ ] Build generation API: POST `{"class": "cat", "steps": 100}` → return generated PNG
- [ ] **Ship:** `generative/` + `generative/gen_api/` + `results/generated_samples/`

---

## Interview Topics

- **Why is the reparameterization trick necessary?** Sampling is not differentiable — you can't backprop through a sample. Reparameterization moves the randomness outside the computational graph: `z = μ + σ·ε`, `ε ~ N(0,1)`. Gradients flow through μ and σ.
- **What is mode collapse in GANs?** Generator produces only a few distinct outputs regardless of input noise — ignores most of the data distribution. Causes: discriminator becomes too strong, minimax instability.
- **Why does DDPM work better than GANs?** Stable training — no adversarial game. Iterative refinement — each step is a small, learnable denoising. Can model complex multimodal distributions.
- **What is classifier-free guidance?** Train a single conditional model with and without the condition (randomly drop condition during training). At inference: `output = uncond_output + scale * (cond_output - uncond_output)`. Higher scale = stronger adherence to condition.

---

## Definition of Done

- [ ] VAE generates recognizable face samples from prior, interpolation is smooth
- [ ] DCGAN trains without mode collapse — generates diverse 64×64 images
- [ ] DDPM trains and denoising steps are visually progressive (noise → image)
- [ ] FID score computed and decreasing with training epochs
- [ ] API accepts class label, returns generated PNG within 10 seconds
- [ ] All tests pass with `pytest generative/`
