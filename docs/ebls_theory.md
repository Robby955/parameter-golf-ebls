# Empirical Bayes Layer Sharing: Statistical Foundations

## The Stein Paradox in Parameter Space

The James-Stein estimator (Stein 1956; James & Stein 1961) demonstrated a counterintuitive result: when estimating three or more normal means simultaneously, shrinking each individual estimate toward the grand mean **always** reduces total mean squared error, even when the means are unrelated. This was later given a Bayesian interpretation by Efron & Morris (1973) — the shrinkage estimator is the posterior mean under an empirical Bayes model where the individual means are drawn from a shared prior.

$$\hat{\theta}_i^{JS} = \bar{\theta} + \left(1 - \frac{(p-2)\sigma^2}{\sum_j (\theta_j - \bar{\theta})^2}\right)(\theta_i - \bar{\theta})$$

The key insight is that **borrowing strength** across related estimation problems improves every individual estimate. The shrinkage factor is learned from the data itself — hence "empirical Bayes."

## Application to Transformer Layer Weights

In a transformer with $L$ layers, each layer $i$ has weight matrices $W_i \in \mathbb{R}^{d \times d}$. Under full independence (standard architecture), these are estimated separately. Under full sharing (recurrent architecture), all layers use identical weights $W_{\text{shared}}$.

EBLS occupies the continuous spectrum between these extremes. We decompose each layer's effective weights as:

$$W_i = W_{\text{shared}} + \gamma_i \cdot \Delta_i$$

where:
- $W_{\text{shared}}$ is the shared prototype (analogous to the grand mean $\bar{\theta}$)
- $\Delta_i = A_i B_i$ is a low-rank deviation (analogous to $\theta_i - \bar{\theta}$)
- $\gamma_i = \sigma(\text{logit}_i) \in [0, 1]$ is the learned shrinkage factor

The shrinkage factor $\gamma_i$ plays the role of $1 - \frac{(p-2)\sigma^2}{\|\theta - \bar{\theta}\|^2}$ in James-Stein: it controls how much each layer's weights deviate from the shared prototype. When $\gamma_i \to 0$, layer $i$ is fully tied to the prototype. When $\gamma_i \to 1$, it operates independently (up to the LoRA rank constraint).

## Why This Analogy is Imperfect (and Why It Works Anyway)

The formal James-Stein result assumes:
1. Gaussian observations with known variance
2. Quadratic loss
3. Independent estimation problems

None of these hold for neural network weights. The weights are not observed but optimized; the loss is cross-entropy, not quadratic; and layer weights interact through the forward pass.

However, the **intuition** transfers robustly:

**Shared structure exists.** Transformer layers perform similar computations (attention + MLP), and adjacent layers often learn similar features. The weight matrices are not arbitrary — they inhabit a low-dimensional manifold in parameter space. Sharing a prototype and learning deviations respects this structure.

**Capacity is scarce.** Under the 16MB artifact limit, every parameter must earn its place. If two layers truly need different weights, the model will learn $\gamma_i > 0$ and pay the storage cost. If they don't, shrinkage regularization drives $\gamma_i \to 0$ and the LoRA deviation is effectively pruned. This is analogous to the bias-variance tradeoff that makes James-Stein work: sharing reduces variance at the cost of bias, and the data determines the optimal tradeoff.

**The empirical Bayes mechanism is implicit.** The shared weights $W_{\text{shared}}$ are trained jointly with all virtual layers that reference them. The gradient signal from all virtual layers shapes the prototype, making it a good "prior" even without explicit Bayesian modeling. This mirrors how empirical Bayes estimates the prior from the data rather than specifying it a priori.

## Empirical Evidence: What the Gammas Tell Us

Our experiments consistently show:

| Component | Gamma Pattern | Interpretation |
|---|---|---|
| MLP weights | $\gamma \approx 0.000$ for all layers | Full sharing is optimal; MLP computation is layer-invariant |
| Attention weights | $\gamma \approx 0.004$ for early layers, $\approx 0$ for later layers | Minimal specialization; attention is nearly layer-invariant too |

This finding has architectural implications. It suggests that in tiny transformers under compression constraints:

1. **MLP weight sharing is essentially free** — the model discovers this automatically rather than requiring manual architecture search
2. **Attention needs barely any per-layer specialization** — the shared attention patterns capture most of what each virtual layer needs
3. **The LoRA rank (8) is much more than sufficient** — with gammas near zero, even rank 1 would suffice

These conclusions are robust across different model widths (640, 768, 1024) and training durations (500–4572 steps).

## Connection to Other Layer-Sharing Approaches

Several concurrent approaches in the challenge also exploit layer sharing:

- **Hard recurrence** (XSA4): Manually specifies which layers share weights. EBLS learns this from data.
- **Universal Transformer** (Dehghani et al., 2019): Shares all layers with per-step halting. EBLS allows per-layer, per-component sharing decisions.
- **Cross-layer parameter sharing** (ALBERT; Lan et al., 2020): Shares all layers uniformly. EBLS learns non-uniform sharing.

EBLS is unique in providing a **continuous, learned** sharing spectrum with a principled regularizer. The shrinkage penalty has a clear statistical motivation: encourage sharing unless the data provides evidence for deviation.

## References

- Efron, B., & Morris, C. (1973). Stein's Estimation Rule and Its Competitors — An Empirical Bayes Approach. *Journal of the American Statistical Association*, 68(341), 117–130.
- James, W., & Stein, C. (1961). Estimation with Quadratic Loss. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361–379.
- Stein, C. (1956). Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution. *Proceedings of the Third Berkeley Symposium*, 1, 197–206.
- Dehghani, M., et al. (2019). Universal Transformers. *ICLR 2019*.
- Lan, Z., et al. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. *ICLR 2020*.
