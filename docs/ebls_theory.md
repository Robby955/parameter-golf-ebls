# Empirical Bayes Layer Sharing: Statistical Foundations

## 1. The Stein Phenomenon and Shrinkage Estimation

In 1956, Charles Stein proved a result that upended classical statistical thinking: when estimating three or more normal means simultaneously, the maximum likelihood estimator (which treats each mean independently) is *inadmissible* under squared-error loss (Stein, 1956). James and Stein (1961) gave an explicit dominating estimator:

$$\hat{\theta}_i^{JS} = \bar{\theta} + \left(1 - \frac{(p-2)\sigma^2}{\sum_{j=1}^{p} (\theta_j - \bar{\theta})^2}\right)(\theta_i - \bar{\theta})$$

The shrinkage factor $\left(1 - \frac{(p-2)\sigma^2}{\|\theta - \bar{\theta}\|^2}\right)$ is data-dependent: it pulls each estimate toward the grand mean, with the strength of shrinkage determined by the observed dispersion. When the individual means are close together (low dispersion), the estimator shrinks aggressively; when they are far apart, it barely shrinks at all.

Efron and Morris (1973) showed this has a natural Bayesian interpretation. Under the hierarchical model

$$\theta_i \sim \mathcal{N}(\mu, A), \quad X_i \mid \theta_i \sim \mathcal{N}(\theta_i, \sigma^2),$$

the posterior mean $\mathbb{E}[\theta_i \mid X]$ takes the form of a shrinkage estimator, with the shrinkage factor determined by the ratio of within-group variance $\sigma^2$ to total variance $\sigma^2 + A$. The James-Stein estimator approximates this posterior mean by estimating the hyperparameter $A$ from the data — the defining move of empirical Bayes.

The crucial insight is that **borrowing strength** across related estimation problems improves *every* individual estimate. This holds even when the parameters being estimated are, in truth, unrelated. It is a consequence of the bias-variance tradeoff operating in high dimensions: the variance reduction from shrinkage outweighs the bias introduced.

## 2. From Shrinkage to Layer Sharing

Consider a transformer with $L$ layers, each parameterized by weight matrices $\{W_i\}_{i=1}^{L}$. Two extreme architectures correspond to two extreme estimation strategies:

- **Independent layers** (standard transformer): Each $W_i$ is estimated separately. No information is shared. This is the MLE.
- **Full sharing** (recurrent transformer): All layers use $W_{\text{shared}}$. Maximum shrinkage — each layer is forced to the grand mean.

EBLS occupies the continuous spectrum between these extremes. We decompose each layer's effective weight matrix as:

$$W_i = W_{\text{shared}} + \gamma_i \cdot \underbrace{A_i B_i}_{\text{rank-}r\text{ deviation}}$$

where:
- $W_{\text{shared}} \in \mathbb{R}^{d \times d}$ is the shared prototype, playing the role of the grand mean $\bar{\theta}$
- $A_i B_i$ is a rank-$r$ deviation, analogous to the individual departure $\theta_i - \bar{\theta}$
- $\gamma_i = \sigma(\text{logit}_i) \in [0, 1]$ is a learned shrinkage factor

The shrinkage factor $\gamma_i$ governs the bias-variance tradeoff for each layer. When $\gamma_i \to 0$, layer $i$ is fully tied to the prototype (maximum bias, minimum variance). When $\gamma_i \to 1$, it deviates freely within the rank-$r$ subspace (minimum bias, maximum variance). The sigmoid parameterization ensures $\gamma_i \in [0, 1]$ and provides a smooth gradient landscape.

We encourage sharing with a regularization penalty:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i=1}^{L} \gamma_i$$

This $\ell_1$-type penalty on the shrinkage factors is the direct analogue of the Bayesian prior encouraging concentration around the group mean. Under the empirical Bayes interpretation, $\lambda$ plays the role of a precision hyperparameter on the prior for deviations: large $\lambda$ corresponds to a strong prior belief that layers should be similar.

## 3. Why the Analogy is Inexact but Operative

The formal conditions of the Stein result — Gaussian observations, quadratic loss, known variance — do not hold for neural network weights. The weights are not "observed" but iteratively optimized; the loss is cross-entropy; the layers interact through the forward pass rather than being estimated independently.

Three properties of the problem make the shrinkage intuition transfer nonetheless:

**Shared structure.** Transformer layers perform the same computation (multi-head attention followed by a feedforward network) with the same input/output dimensionality. Empirically, learned weight matrices across layers occupy a low-dimensional subspace of $\mathbb{R}^{d \times d}$. This is the neural network analogue of the hierarchical model: the layer weights are "drawn from" a common generative process determined by the architecture and training distribution.

**Scarce capacity.** Under the 16MB artifact constraint, every parameter must justify its inclusion. The shrinkage framework makes this tradeoff explicit: a layer deviates from the prototype only if the deviation reduces loss more than the regularization penalty. Parameters "freed" by sharing (when $\gamma_i \approx 0$) can be reallocated to increase model width, which benefits all layers uniformly.

**Implicit empirical Bayes.** The shared weights $W_{\text{shared}}$ receive gradient signal from every virtual layer that references them. This joint optimization makes the prototype a data-driven estimate of the "typical" layer computation — an empirical prior. Unlike classical empirical Bayes where the prior is estimated in a separate step, here the prior and the individual estimates are optimized simultaneously. This is closer to the fully Bayesian treatment, but without the computational cost of posterior inference.

## 4. Interpreting Gamma Convergence

After training on 8×H100 (4572 steps, 10-minute wallclock), the learned shrinkage factors show a striking pattern:

| Virtual Layer | Attention $\gamma$ | MLP $\gamma$ |
|:---:|:---:|:---:|
| 0 | 0.0046 | 0.0006 |
| 1 | 0.0029 | 0.0000 |
| 2 | 0.0018 | 0.0000 |
| 3–8 | ≤ 0.0006 | 0.0000 |

**MLP weights converge to full sharing** ($\gamma \approx 0$ for all layers). The model discovers, through gradient-based optimization of the shrinkage factors, that MLP computation should be identical across all virtual layers. This is a nontrivial finding: it says that the feedforward network's role — expanding activations into a higher-dimensional space, applying a nonlinearity, and projecting back — does not need to vary with depth in a 9-layer model under compression constraints.

In the language of empirical Bayes: the posterior variance of MLP weights across layers is negligible relative to the shared component. The "true" MLP parameters are so similar that shrinkage to the mean incurs effectively zero bias.

**Attention shows minimal, monotonically decaying specialization.** Early layers have $\gamma_{\text{attn}} \approx 0.004$ — barely nonzero. This suggests that early-layer attention patterns are slightly more position-dependent (perhaps because they must establish initial token interactions that later layers can refine), but the magnitude of deviation is two orders of magnitude below the maximum allowed ($\gamma = 1$).

**Architectural implications.** This gamma pattern provides empirical evidence for design choices that other challenge submissions make heuristically:
1. MLP weight sharing is essentially free — EBLS discovers this automatically
2. A recurrent architecture (where all layers share weights) loses very little for MLP computation
3. The LoRA rank of 8 is far more than necessary — with gammas near zero, even rank 1 would suffice

These findings are robust across model widths (640, 768, 1024 dimensions) and training durations (500–4572 steps), suggesting they reflect genuine structure in the problem rather than optimization artifacts.

## 5. Relation to Other Approaches

| Method | Sharing Structure | Shrinkage | Data-Driven |
|---|---|---|---|
| Standard transformer | None (independent layers) | — | — |
| ALBERT (Lan et al., 2020) | Uniform full sharing | Hard ($\gamma = 0$) | No |
| Universal Transformer (Dehghani et al., 2019) | Full sharing + per-step halting | Hard | Halting only |
| XSA4 / hard recurrence | Manual layer groups | Hard | No |
| **EBLS** | **Per-layer, per-component** | **Continuous, learned** | **Yes** |

EBLS is distinguished by providing a continuous, learned sharing spectrum with a regularizer that has a clear statistical motivation. The model is free to discover any sharing pattern — from full independence to full sharing — without architectural search.

## 6. Validating Hard-Sharing Decisions: The XSA4 Connection

The gamma convergence results have a direct connection to other entries on the Parameter Golf leaderboard. XSA4 and similar hard-recurrence submissions manually choose to share all layer weights, making this decision by intuition or grid search. EBLS arrives at the same conclusion through optimization: the learned shrinkage factors converge to near-zero, meaning the model independently discovers that full (or near-full) sharing is optimal under the 16MB constraint.

This is more than a confirmation of existing practice. The component-level granularity of EBLS reveals structure that hard sharing cannot: MLP weights converge to $\gamma = 0.0000$ while attention weights retain $\gamma \approx 0.004$ in early layers. A hard-sharing architecture treats both components identically. EBLS shows that the optimal sharing pattern is not uniform — MLP truly wants identical weights across layers, while attention benefits from a trace of per-layer specialization. This suggests that future architectures could achieve slightly better results by sharing MLP weights fully while allowing minimal attention deviation, a design point that hard recurrence cannot express.

More broadly, the gamma convergence pattern constitutes empirical evidence about transformer layer structure that generalizes beyond this particular challenge. Under compression constraints, depth-wise weight variation is overwhelmingly concentrated in shared structure rather than per-layer specialization. The fact that gradient-based optimization of continuous shrinkage factors converges to the same qualitative answer as manual architecture search — but with finer component-level resolution — validates both the EBLS framework and the intuitions underlying hard-sharing approaches.

## References

- Efron, B. & Morris, C. (1973). Stein's Estimation Rule and Its Competitors — An Empirical Bayes Approach. *Journal of the American Statistical Association*, 68(341), 117–130.
- James, W. & Stein, C. (1961). Estimation with Quadratic Loss. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 361–379.
- Stein, C. (1956). Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution. *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 197–206.
- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J. & Kaiser, Ł. (2019). Universal Transformers. *ICLR 2019*.
- Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P. & Soricut, R. (2020). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. *ICLR 2020*.
