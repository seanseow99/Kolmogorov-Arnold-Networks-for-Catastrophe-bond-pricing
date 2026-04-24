# CAT Bond Pricing with Kolmogorov-Arnold Networks

**NUS UROPS | Supervised by Prof. Julian Sester | April 2026**

This repository implements an interpretable machine-learning surrogate for catastrophe (CAT) bond valuation using **Kolmogorov-Arnold Networks (KANs)**. The objective is to approximate Monte Carlo CAT bond prices with high accuracy while extracting a compact symbolic pricing formula that can be inspected, evaluated quickly, and checked against financial monotonicity constraints.

## Overview

Catastrophe bonds are insurance-linked securities whose cash flows depend on whether cumulative catastrophe losses exceed a contractual trigger threshold. Standard valuation under compound Poisson loss models generally requires Monte Carlo simulation because the trigger probability is path-dependent and does not admit a simple closed-form solution in most settings. Existing neural-network approaches can learn this pricing map efficiently, but standard feedforward networks remain largely black-box models. Neural-network CAT bond valuation has been proposed as a fast and accurate alternative to Monte Carlo and PIDE methods, with inputs reflecting market conditions and contract features. :contentReference[oaicite:0]{index=0}

This project extends that idea by replacing a black-box feedforward neural network with a **KAN-based residual surrogate**. KANs place learnable univariate functions on edges rather than fixed activation functions on nodes, allowing trained networks to be simplified, pruned, and converted into symbolic expressions. :contentReference[oaicite:1]{index=1}

The final model is designed to satisfy three goals:

1. **Accuracy**: match Monte Carlo CAT bond prices on unseen parameter configurations.
2. **Interpretability**: extract a closed-form symbolic approximation.
3. **Financial consistency**: preserve expected monotonicity with respect to interest rates, loss intensity, and trigger threshold.

## Model Setup

The underlying pricing model follows a reduced-form CAT bond framework:

- Aggregate catastrophe losses follow a **compound Poisson process**
  
  $$
  L_t = \sum_{i=1}^{M_t} X_i,
  $$

  where \(M_t \sim \mathrm{Poisson}(\lambda t)\), and severities \(X_i\) are lognormally distributed.

- A trigger occurs when cumulative losses exceed a threshold \(D\):

  \[
  \tau = \inf \{t : L_t \ge D\}.
  \]

- Interest rates follow a **Vasicek short-rate model**, giving an affine zero-coupon discount factor.

- CAT bond prices are computed by discounting coupon and principal cash flows conditional on survival of the loss trigger.

The project focuses on pricing maps of the form

\[
C = C(r_0, \lambda, D, N, T),
\]

where:

| Variable | Meaning |
|----------|---------|
| \(r_0\) | Initial short rate |
| \(\lambda\) | Catastrophe arrival intensity |
| \(D\) | Trigger threshold |
| \(N\) | Coupon frequency |
| \(T\) | Maturity |

## Baseline-Plus-Residual Architecture

Instead of learning the full price directly, the model uses a **baseline-plus-residual decomposition**.

First, an analytical approximation \(C_{\text{base}}\) is computed using:

- Vasicek zero-coupon bond pricing,
- compound Poisson moment matching,
- a lognormal approximation to aggregate losses,
- survival probabilities at coupon dates.

The KAN then learns the residual correction in log-ratio form:

\[
\ell(x) = \log \frac{C_{\text{MC}}(x)}{C_{\text{base}}(x) + \varepsilon}.
\]

The final reconstructed price is

\[
\widehat C(x)
=
\left(C_{\text{base}}(x) + \varepsilon\right)
\exp(\widehat \ell_{\text{KAN}}(x)).
\]

This stabilizes training because the KAN only needs to learn the deviation between the analytical baseline and the Monte Carlo price, rather than the entire pricing function.

## Kolmogorov-Arnold Network Surrogate

The KAN is trained on standardized input features and Monte Carlo-generated target prices. A KAN layer has the form

\[
x_{l+1,j}
=
\sum_i \phi_{l,j,i}(x_{l,i}),
\]

where each \(\phi_{l,j,i}\) is a learnable univariate spline function. Unlike an MLP, which uses scalar weights and fixed nonlinearities, the KAN learns nonlinear edge functions directly.

The training pipeline is:

1. Train spline-based KAN on the residual target.
2. Apply sparsity and entropy regularization.
3. Prune weak edges/nodes.
4. Refine the spline grid.
5. Convert selected spline functions into symbolic functions.
6. Refit symbolic affine parameters.
7. Evaluate the final symbolic pricing formula.

Candidate symbolic functions include simple financially interpretable functions such as:

```python
["x", "x^2", "x^3", "exp", "Phi"]

