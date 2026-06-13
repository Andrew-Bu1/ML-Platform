# Sprint 22 — Capstone D: Controllable Diffusion Generation

**Theme:** Conditional DDPM, classifier-free guidance, CLIP-guided generation
**Duration:** 3 weeks
**Phase:** Phase 7 — Capstone Systems

---

## Goal

Extend your DDPM from Sprint 15 into a controllable generation system. Users describe what they want (class label or text) and receive a generated image. Evaluate with FID and CLIP score. Ship a streaming generation UI.

---

## Concepts

### Class-Conditional DDPM

- Embed class label c into a vector: `c_emb = Embedding(num_classes, d_model)`.
- Inject into UNet: add `c_emb` to time embedding before projecting into each ResBlock.
- At inference: condition on desired class label.

### Classifier-Free Guidance (CFG)

- During training: randomly set condition to `null` with probability p_uncond=0.1.
- Model learns both conditional and unconditional distributions from one network.
- At inference: `output = uncond_pred + guidance_scale * (cond_pred - uncond_pred)`.
- Higher guidance scale → stronger adherence to condition, less diversity.

### CLIP-Guided Diffusion

- At each diffusion step: after predicting x_0 from x_t, compute CLIP similarity between x_0 and text prompt.
- Use gradient of CLIP similarity to nudge x_t toward higher alignment.
- No additional training needed — CLIP gradients guide existing diffusion model.

### Evaluation

- **FID (Fréchet Inception Distance):** compares real vs generated image distributions in Inception feature space. Lower is better.
- **CLIP Score:** cosine similarity between generated image CLIP embedding and prompt CLIP embedding. Higher means better text-image alignment.
- **IS (Inception Score):** measures quality and diversity. Less commonly used now.

---

## Tasks

### Conditional Model

- [ ] Extend UNet with class embedding injection via AdaGroupNorm or addition to time embedding
- [ ] Retrain on CIFAR-10 with class conditioning — verify conditional FID < unconditional FID
- [ ] Implement class label dropout for CFG training (p_uncond=0.1)
- [ ] Implement CFG sampling: run denoiser twice per step (cond + uncond), interpolate
- [ ] Sweep guidance scales [1, 2, 5, 10] — show FID vs sample diversity tradeoff
- [ ] **Ship:** `generative/conditional_ddpm.py` + `generative/cfg.py`

### CLIP Guidance

- [ ] Load CLIP (use OpenAI CLIP weights via HuggingFace) — freeze all weights
- [ ] Implement CLIP-guided diffusion step: add `∇_x [CLIP_similarity(x_0_hat, prompt)]` to score
- [ ] Test: prompt "a red car" should steer generation toward red-tinted samples
- [ ] **Ship:** `generative/clip_guided_diffusion.py`

### Evaluation

- [ ] Implement FID: extract Inception features for real and generated sets, compute Fréchet distance
- [ ] Implement CLIP score: embed generated images and prompts, compute cosine similarities
- [ ] Generate 1000 samples per class, compute class-conditional FID
- [ ] **Ship:** `generative/gen_eval.ipynb` + `results/generation_eval.json`

### Generation UI

- [ ] Build React UI: text input + class selector + guidance scale slider
- [ ] FastAPI SSE endpoint: stream intermediate diffusion steps as JPEG frames
- [ ] Display: show denoising animation from noise to image in real time
- [ ] Add download button for final generated image
- [ ] **Ship:** `generative/gen_ui/` (React) + FastAPI SSE backend

---

## Definition of Done

- [ ] Conditional FID < 50 on CIFAR-10 (unconditional baseline should be > 100)
- [ ] CFG at scale=7.5 visually produces cleaner, more class-specific images than scale=1
- [ ] CLIP-guided generation shows measurable CLIP score improvement vs unguided
- [ ] Streaming UI displays denoising animation correctly in browser
- [ ] Evaluation notebook documents FID + CLIP scores with charts
