# Plan: Moran Model for MERFISH Spatial Deconvolution

## 1. What Count Bridges Does for MERFISH

### The Task
**Spatial deconvolution**: Given bulk gene expression $X_0 \in \mathbb{Z}_{\geq 0}^{649}$ at a tissue spot (the sum of all cells), recover the individual cell-level profiles $x_0^{(1)}, \ldots, x_0^{(N)} \in \mathbb{Z}_{\geq 0}^{649}$ such that $\sum_i x_0^{(i)} = X_0$.

### Data Structure
- **2 tissue sections** (S1R1, S1R2) from Vizgen MERFISH Mouse Brain
- **~9,148 tissue spots** (spatial locations), each containing a variable number of cells
- **~165,918 cells total**, average ~18 cells per spot
- Each cell has:
  - Gene expression count vector: $x \in \mathbb{Z}_{\geq 0}^{649}$ (649 genes)
  - DAPI nuclear image: $256 \times 256$ grayscale
- Each spot has:
  - Bulk expression: $X_0 = \sum_{i \in \text{spot}} x_0^{(i)}$ (649-dim integer vector)
  - Variable number of cells (handled via sparse aggregation matrix)

### How Count Bridges Treats Each Cell

**The Skellam bridge operates per-cell, per-gene, independently.** Each of the 649 gene dimensions for each cell is an independent 1D birth-death bridge:

- $x_0^{(i,j)}$ = true count of gene $j$ in cell $i$
- $x_1^{(i,j)} \sim |\mathcal{N}(0, 10)|$ = noisy source count
- The Skellam bridge interpolates: $x_t^{(i,j)} = x_0^{(i,j)} + B_t - D_t$ (births minus deaths on $\mathbb{Z}$)
- Bridge conditional $P(x_t | x_0, x_1)$ has exact closed form (Binomial + Hypergeometric + Bessel)

**There is no interaction between cells.** Cell $i$ and cell $j$ at the same spot are bridged completely independently. The only coupling is through the **aggregation loss**: the model's predictions must sum to the observed bulk $X_0$.

### Architecture: MultimodalUViT

The model processes each cell independently:
- Input: $(x_t^{(i)}, t, \text{noise}, \text{DAPI}_i)$ for each cell
- 649D count vector $\to$ 4 patch embeddings (768-dim each)
- 256$\times$256 DAPI image $\to$ 256 patch embeddings (via ViT)
- 10-layer Transformer (768-dim, 12 heads) processes 262 tokens
- Output: predicted count vector $\hat{x}_0^{(i)} \in \mathbb{R}_{\geq 0}^{649}$ (Softplus)

### Training: E-M Loop

Each epoch alternates:
1. **E-step**: Sample $x_0$ from the current model via reverse Skellam bridge, conditioned on $X_0$ (aggregation constraint). Uses IPF rescaling + randomized rounding to ensure $\sum_i x_0^{(i)} = X_0$ exactly.
2. **M-step**: Train on $(x_0^{\text{inferred}}, x_1)$ pairs via the Skellam bridge + energy score loss. The loss is computed at the **group level** after aggregating cell predictions: $L = \text{EnergyScore}(A \cdot \hat{x}_0, X_0)$.

### Aggregation Mechanism

The sparse aggregation matrix $A \in \{0,1\}^{G \times C}$ maps cells to spots:
$$A_{g,c} = \begin{cases} 1 & \text{if cell } c \text{ belongs to spot } g \\ 0 & \text{otherwise} \end{cases}$$

- Forward: $A \cdot \hat{x}_0 \in \mathbb{Z}^{G \times 649}$ gives predicted bulk expression per spot
- The energy score loss compares $A \cdot \hat{x}_0$ to $X_0$ (the observed bulk)
- At inference: IPF rescaling ensures $A \cdot \hat{x}_0 = X_0$ exactly

### Evaluation Metrics
- Energy distance (OT-based)
- Sliced Wasserstein distance (100 random projections)
- MMD with RBF kernel
- MSE
- Statistical moments: mean, variance, skewness, kurtosis errors
- Covariance Frobenius norm
- Cell type proportions via k-NN on PCA-reduced counts

---

## 2. What the Moran Model Provides

### The Key Limitation of Count Bridges for MERFISH

Count Bridges generates each cell **independently**. The only coupling is the aggregation constraint, enforced post-hoc via IPF rescaling. This means:

1. Two adjacent cells at the same spot can receive wildly different predicted profiles, even if they are clearly the same cell type
2. There is no notion of cell type clustering -- the model doesn't know that cells come in discrete types
3. The aggregation constraint is a hard projection, not part of the generative model itself

### What Moran Adds: Inter-Cell Coupling

In the Moran framework for MERFISH:
- **Individuals** = cells within a tissue spot
- **Type** of individual $i$ = gene expression profile $x^{(i)} \in \mathbb{Z}_{\geq 0}^{649}$
- **Mutation** = stochastic changes to gene expression (birth-death per gene, or random perturbation)
- **Resampling** = cell $i$ copies cell $j$'s expression profile, weighted by a spatial kernel $K(i, j)$

The Moran resampling creates **inter-cell coupling**: nearby cells tend to converge to similar expression profiles. This naturally produces **cell type clusters** as a mathematical consequence of the Donnelly-Kurtz lookdown construction.

### The De Finetti Prior

At stationarity, the Moran model's empirical measure converges to $\text{DP}(\theta, \nu)$ where:
- $\theta = 2\gamma_{\text{mut}} / \kappa$ controls the number of distinct cell types
- $\nu$ is the stationary distribution of the mutation process on $\mathbb{Z}_{\geq 0}^{649}$

This means the **prior already has cell type structure**: the DP clustering produces groups of cells sharing the same expression profile (atoms of the DP), with the number of groups controlled by $\theta$.

Compare to Count Bridges' source distribution: $x_1^{(i)} \sim |\mathcal{N}(0, 10)|$ i.i.d. -- no cell type structure at all.

---

## 3. Choice of Kernel

The resampling kernel $K(i, j)$ determines which cells influence each other. Three options:

### Option A: Spatial Kernel (Recommended)
$$K(i, j) = \exp\left(-\frac{\|p_i - p_j\|^2}{2h^2}\right)$$
where $p_i, p_j$ are the spatial positions (centroids) of cells $i, j$ within the tissue spot.

**Advantages:**
- Biologically motivated: spatial proximity $\to$ shared lineage, shared microenvironment, paracrine signaling
- Fixed and precomputable (no chicken-and-egg problem)
- Cell positions are derived from the DAPI images (already in the dataset)
- Captures exactly what makes Moran meaningful: the genealogical structure maps onto the lookdown construction

**Challenge:** Cell positions within a spot need to be extracted. The current dataset stores DAPI images per cell but not explicit $(x, y)$ coordinates within the spot. These may need to be computed from the original MERFISH data, or approximated from image centroids.

### Option B: DAPI Morphology Kernel
$$K(i, j) = \exp\left(-\frac{\|f(I_i) - f(I_j)\|^2}{2h^2}\right)$$
where $f(I)$ extracts CNN features from DAPI image $I$.

**Advantages:** Cells with similar nuclear morphology are often the same type.
**Disadvantages:** Requires a pretrained feature extractor; adds complexity.

### Option C: Expression Similarity Kernel
$$K(i, j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2h^2}\right)$$

**Disadvantages:** Creates state-dependent rates in the forward process, which complicates training. The kernel changes as expression changes. Better used only at initialization (from clean data) rather than dynamically.

**Recommendation:** Start with Option A (spatial kernel). It's the most principled and the simplest to implement.

---

## 4. Key Implementation Questions

### Q1: How are cell positions stored?

The `S1R1.npz` files contain more fields than the dataset loader uses:

```
imgs, counts, spots, x_um, y_um, annotations, visium_simulated, cell_type_counts, n_cells
```

**Spatial coordinates ARE available**: `x_um` and `y_um` give cell centroid positions in micrometres. The dataset loader (`merfish_deconv.py`) only reads `imgs`, `counts`, and `spots` — it ignores spatial coordinates entirely. A proper spatial kernel $K(i,j) = \exp(-\|p_i - p_j\|^2 / 2h^2)$ can be built immediately using `(x_um, y_um)`.

Additionally, `annotations` contains cell type labels at various Leiden clustering resolutions, and `visium_simulated` likely contains simulated Visium-style bulk measurements.

## Count Bridges Benchmark Results

The evaluation compares three quantities per spot:
- **Pred vs True**: Generated profiles vs MERFISH ground truth (primary metric)
- **Pred vs Spot Mean**: Generated profiles vs naive baseline (assign every cell the spot mean)
- **True vs Spot Mean**: Inherent biological variability (oracle ceiling)

### S1R1 Results (4,358 spots, 73,613 cells)

| Metric | Pred vs True | Pred vs Spot Mean | True vs Spot Mean |
|--------|-------------|-------------------|-------------------|
| Energy Distance | 8.891 | 42.903 | 41.717 |
| Sliced Wasserstein | 0.017 | 0.034 | 0.030 |
| MMD (RBF) | 0.203 | 0.419 | 0.409 |

**Interpretation:**
- Count Bridges works well: Pred vs True (8.9) is much smaller than the inherent variability (41.7)
- Variability is correctly captured: Pred vs Spot Mean ≈ True vs Spot Mean
- **This is the target to beat.** Our Moran model must improve Pred vs True while maintaining correct variability.

### Q2: How does variable N per spot interact with the Moran model?

Different spots have different numbers of cells (~3 to ~50+). The Moran model needs a fixed population size $N$ within each simulation.

**Options:**
1. **Pad to max N**: Pad smaller spots with dummy cells, mask in loss. Simple but wasteful.
2. **Group by similar N**: Batch spots with similar cell counts together. Efficient but complex.
3. **Use the actual variable N**: The Moran forward process works for any $N$. Just need the model to handle variable input sizes (Transformer/DeepSets naturally do this).

Option 3 is most natural since the existing collation already handles variable sizes via the sparse aggregation matrix.

### Q3: How does the aggregation constraint work with Moran?

Count Bridges enforces $\sum_i \hat{x}_0^{(i)} = X_0$ via IPF rescaling at inference time. With Moran:

- **During training**: The Moran forward corrupts the population jointly. The aggregation constraint is enforced through the loss (same as Count Bridges).
- **During generation**: Start from DP prior $\to$ denoise $\to$ apply IPF rescaling to ensure sum constraint. The DP prior naturally produces cell type clusters, so the rescaling is less violent (cells within a cluster already have similar profiles).

### Q4: What architecture changes are needed?

Count Bridges' MultimodalUViT processes each cell independently (no inter-cell attention). For Moran, the model needs to see all cells in a spot simultaneously to exploit the inter-cell coupling.

**Options:**
1. **DeepSets wrapper**: Per-cell UViT encoder $\to$ mean-pool across cells $\to$ per-cell decoder with global context. Minimal change to existing architecture.
2. **Cross-attention**: Add cross-attention layers between cells in the ViT. More expressive but heavier.
3. **Separate set-level head**: Keep per-cell UViT for feature extraction, add a lightweight Transformer on top for inter-cell reasoning.

Option 1 is the safest starting point: wrap the existing UViT in a DeepSets framework. Each cell gets its own UViT features, these are pooled to create a global context, then each cell's prediction is conditioned on the context.

### Q5: How does the E-M loop change?

The E-step currently uses the Skellam bridge reverse sampler. With Moran:

- **E-step**: Use the Moran reverse process (iterative denoising from DP prior) instead of the Skellam reverse. The aggregation constraint is applied via IPF rescaling after each denoising step.
- **M-step**: Use the Moran forward process (mutation + resampling) instead of the Skellam bridge. The loss is still energy score at the group level.

The E-M structure remains the same; only the bridge is swapped.

---

## 5. Implementation Plan

### Phase 1: Two Moons N-Sweep (Proof of Concept)

**Status: Framework implemented** (`moran/experiment.py`).

The N-sweep on discrete two moons validates the core mechanism:
- Same data as Count Bridges (value_range=196)
- Same metrics (MMD, W2, coverage, off-support)
- Sweep N (population size) and $\theta$ (mutation-drift ratio)
- Compare aggregated Moran particles against Count Bridges' per-point generation

This proves (or disproves) that inter-particle coupling helps for discrete data generation before scaling to 649 dimensions.

### Phase 2: MERFISH Data Exploration

Before writing code, we need to understand:
1. **Cell positions**: Are they in the original Vizgen data? Can we extract them?
2. **Cell count distribution**: How many cells per spot? (min, max, mean, distribution)
3. **Gene expression structure**: How many effective cell types? What does PCA look like?
4. **Spot diversity**: How different are spots from each other?

This requires access to the actual MERFISH `.npz` files (currently on `/orcd/data/...`).

### Phase 3: Moran Forward for MERFISH

Implement the Moran forward process for 649-dim count vectors:
- **Mutation**: Per-gene immigration-death (or Poisson birth-death) -- similar to what Skellam does but as a CTMC rather than a bridge
- **Resampling**: Cell $i$ copies cell $j$'s 649-dim profile, weighted by spatial kernel
- **Time scaling**: $1/(1-t)^2$ to ensure convergence to de Finetti equilibrium

Key difference from Skellam: the Moran forward is a **joint process over all cells**, not per-cell independent. The resampling creates correlations between cells.

### Phase 4: Architecture Adaptation

Wrap the existing MultimodalUViT in a DeepSets framework:
1. Per-cell UViT features (reuse existing architecture)
2. Mean-pool across cells in spot $\to$ global context
3. Per-cell decoder conditioned on context

This preserves all the existing image+count processing while adding inter-cell awareness.

### Phase 5: Training Loop

Adapt the E-M training loop:
- Replace Skellam bridge with Moran forward/reverse in both E-step and M-step
- Keep the aggregation constraint and IPF rescaling
- Keep the energy score loss at group level

### Phase 6: Evaluation

Same metrics as Count Bridges:
- Per-spot: energy distance, Wasserstein, MMD, MSE
- Global: cell type proportions, moment errors, covariance structure
- Additional (Moran-specific): U-field purity (do lookdown levels correlate with cell types?)

---

## 6. What Moran Provides That Count Bridges Doesn't

| Property | Count Bridges | Moran |
|----------|--------------|-------|
| Cell interaction | None (independent per cell) | Resampling couples nearby cells |
| Cell type structure | None in prior; emerges only through training | DP prior naturally clusters cells into types |
| Aggregation | Hard constraint (IPF projection) | Soft coupling + hard constraint |
| Genealogy | None | Lookdown levels trace cell lineage |
| Spatial coherence | Only through DAPI images | Explicit spatial kernel in dynamics |

The central prediction: **Moran should produce more spatially coherent cell type assignments**, where nearby cells are more likely to be assigned the same type. This is because the resampling mechanism forces nearby cells to converge to similar profiles during the forward process, and the reverse process preserves this structure.

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cell positions not available in `.npz` | Can't build spatial kernel | Use uniform kernel within spot (all cells equally coupled) |
| 649-dim Moran forward too slow | Training bottleneck | Subsample genes; batch-vectorize; use tau-leaping not Gillespie |
| Variable N per spot complicates batching | Slow training | Group by similar N; pad with masks |
| DP prior doesn't match gene expression | Poor generation | Calibrate $\theta$ and mutation stationary $\nu$ to data |
| MultimodalUViT too heavy for DeepSets wrapper | OOM on GPU | Use lightweight set encoder; keep UViT only for feature extraction |
| Two moons N-sweep shows no Moran advantage | Undermines MERFISH case | The 2D toy may be too simple; 649-dim MERFISH is where coupling matters |
