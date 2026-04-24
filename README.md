# Kolmogorov-Arnold-Networks-for-Catastrophe-bond-pricing
\documentclass[11pt,a4paper]{article}
 
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{booktabs}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
 
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
 
\begin{document}
 
\begin{center}
    {\Large\bfseries CAT Bond Pricing with Kolmogorov--Arnold Networks}\\[6pt]
    {\normalsize Sean Seow Cheng Hong}\\[2pt]
    {\small NUS UROPS $\cdot$ Supervised by Prof Julian Sester $\cdot$ April 2026}
\end{center}
 
\medskip
\hrule
\medskip
 
\textbf{Motivation.}
A central challenge in financial modelling is the trade-off between predictive accuracy and
interpretability. Standard neural network surrogates for derivative pricing are fast but opaque ---
they produce predictions without explanation. Kolmogorov--Arnold Networks (KANs) offer a
different paradigm: by placing learnable univariate functions on edges rather than fixed activations
on nodes, they support \emph{symbolic extraction} --- converting a trained network into a compact
closed-form formula. This paper applies KANs to the pricing of catastrophe (CAT) bonds,
fixed-income instruments that transfer insurance tail risk to capital markets. The outstanding CAT
bond market reached \$61.3 billion at end-2025, making fast and interpretable pricing increasingly
important.
 
\textbf{Setup.}
We work within the compound Poisson loss framework of Baryshnikov et al.\ (1998) and Burnecki
and Kukla (2003), with lognormal loss severities and a Vasicek interest rate model, following the
surrogate modelling approach of Sester and Xu (2025). Our dataset consists of 100,000 Monte Carlo
prices with importance sampling over a grid of parameters $(r_0, \lambda, D, N, T)$. From this we
draw a working sample of 10,000 observations for model development, reserving the remaining 90,000
as a fully disjoint holdout.
 
\textbf{Methodology.}
The core contribution is a \emph{baseline-plus-residual} pipeline. Rather than learning the raw
price, we compute a closed-form lognormal baseline $P_{\text{base}}$ via moment-matched survival
probabilities and Vasicek discounting, then train a shallow KAN $[5,6,1]$ on the log-ratio residual
$y = \log(P/P_{\text{base}})$. This reduces the learning target to a small, smooth function, making
symbolic extraction tractable. Hyperparameters are selected via Tree-structured Parzen Estimation
(TPE) over 100 trials. The symbolic extraction pipeline prunes weak edges, refines the spline grid,
and locks each active edge to a function from the library
$\mathcal{L}_{\text{sym}} = \{x,\, x^2,\, x^3,\, \exp,\, \Phi\}$.
The final model is selected by the combined score
$0.8\cdot R^2_{\text{sym}} + 0.2\cdot R^2_{\text{KAN}}$.
 
\textbf{Theoretical contributions.}
We prove that the true CAT bond price is nonincreasing in the catastrophe arrival intensity
$\lambda$ and the initial short rate $r_0$, and nondecreasing in the trigger threshold $D$,
extending the continuity analysis of Sester and Xu to directional comparative statics. We derive
sufficient conditions on KAN edge functions that guarantee the learned model inherits these
monotonicities, and prove that a monotonicity-penalised training objective converges to the
constrained problem as penalty weights grow.
 
\textbf{Results.}
The extracted symbolic formula consists of a linear term in the standardised inputs and a single
$\Phi$ correction --- a structure that directly reflects the lognormal survival probability in the
baseline. On the disjoint 90,000-observation holdout, the formula achieves an average relative
pricing error of \textbf{0.483\%}, comparable to the pre-symbolic KAN (0.403\%) and the
feedforward neural network of Sester and Xu (0.391\%). Sensitivity analysis confirms that the
symbolic formula satisfies all three comparative statics empirically across the full parameter
range, whereas the pre-symbolic KAN exhibits local monotonicity violations. The formula is also
the only model that produces an analytically tractable closed-form expression suitable for
stress-testing and financial interpretation.
 
\textbf{Conclusion.}
Symbolic KAN surrogates occupy a practical position in the accuracy-interpretability tradeoff for
CAT bond valuation: they sacrifice a small amount of predictive accuracy relative to black-box
networks, but return a compact closed-form pricing formula that can be evaluated instantly,
verified against structural financial properties, and interpreted without the need for post-hoc
explainability tools.
 
\medskip
\hrule
\medskip
 
{\small
\textbf{Keywords:} catastrophe bonds, Kolmogorov--Arnold Networks, symbolic regression,
compound Poisson process, Vasicek model, interpretable machine learning, surrogate pricing.
}
 
\end{document}
