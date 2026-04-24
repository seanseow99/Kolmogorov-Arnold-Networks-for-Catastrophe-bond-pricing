#CAT Bond Pricing with Kolmogorov-Arnold Networks
#NUS UROPS | Supervised by Prof Julian Sester | April 2026
##What is this?
A fast, interpretable pricing model for catastrophe (CAT) bonds — fixed-income instruments that transfer insurance disaster risk to capital markets. The global CAT bond market hit $61.3B at end-2025, making efficient pricing increasingly important.
The core idea: instead of a black-box neural network, can we extract a compact closed-form pricing formula that is both accurate and interpretable?
Methods

Built a baseline-plus-residual pipeline — compute a closed-form baseline price analytically, then train a KAN on the small residual correction only
Used Kolmogorov-Arnold Networks (KANs) — a new neural network architecture that places learnable functions on edges, enabling symbolic extraction into human-readable formulas
Underlying model: compound Poisson loss framework with lognormal severities and Vasicek interest rates
Trained on 10,000 Monte Carlo prices, validated on a disjoint holdout of 90,000

#Results

0.483% average pricing error on 90,000 holdout observations
Comparable accuracy to standard feedforward networks — but returns a closed-form formula instead of a black box
Formula provably satisfies financial monotonicity constraints (standard neural nets violated these)
