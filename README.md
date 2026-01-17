# Statistics & Machine Learning Mini-Projects

A comprehensive collection of mini-projects combining statistical inference, hypothesis testing, and machine learning to solve real-world problems. This repository demonstrates the practical application of statistical rigor in ML workflows, from bias detection to Bayesian modeling.

##  Overview

This repository contains implementations of various projects that bridge the gap between traditional statistical methods and modern machine learning techniques. Each project emphasizes:
- **Rigorous statistical testing** to validate ML findings
- **Uncertainty quantification** through confidence intervals and Bayesian inference
- **Real-world applications** from bias detection to epidemiological modeling

## Project Structure

### Causal Bias Detection in Recommendation Systems
**Goal:** Test whether ML models perpetuate or amplify historical biases

**Components:**
- Binary classification model (Logistic Regression, Random Forest)
- Statistical inference to quantify bias between favored/disadvantaged groups
- Hypothesis testing (χ² test, Z-test) for performance disparities
- Fairness metrics: Disparate Impact & Equal Opportunity tests

**Key Tools:** `scikit-learn`, `statsmodels`, `bootstrap estimation`

**Key Question:** Is the ML model fair, even with good overall performance?

---

###  Feature Importance Robustness Evaluation
**Goal:** Assess the statistical reliability of feature importance scores

**Components:**
- Supervised learning with feature importance (XGBoost, LightGBM)
- Bootstrapping to create confidence intervals for importance scores
- Permutation testing to validate that importance isn't due to chance

**Key Tools:** `XGBoost`, `SHAP`, `ELI5`, `scikit-learn`

**Key Question:** Is the feature importance statistically reliable or just noise?

---

###  Distribution Drift Detection Post-Deployment
**Goal:** Detect and quantify data drift between training and production

**Components:**
- Simulation of deployed model with changing production data
- Non-parametric tests (Kolmogorov-Smirnov, Jensen-Shannon Divergence)
- Confidence intervals for model performance degradation
- Visualization of distribution shifts

**Key Tools:** `scipy.stats`, `numpy`, `bootstrapping`

**Key Question:** Is the model still valid after deployment?

---

###  Bayesian Epidemiological Modeling with Uncertainty Quantification
**Goal:** Estimate true epidemic incidence from incomplete reporting data

**Components:**
- Non-linear Richards growth model for cumulative cases
- Bayesian MCMC inference for posterior distributions
- Highest Density Intervals (HDI) for hidden variables
- Convergence diagnostics (R-hat, ESS) and model comparison (WAIC, LOO-CV)

**Key Tools:** `PyMC3`, `Stan`, `cmdstan`, `ArviZ`

**Key Question:** What's the true epidemic magnitude when reporting is incomplete?

---

###  Regression with Reliable Prediction Intervals
**Goal:** Provide predictions with explicit uncertainty quantification

**Components:**
- Regression models (Linear, Random Forest, Gradient Boosting)
- Non-parametric prediction intervals (quantiles, bootstrap)
- Residual analysis (Shapiro-Wilk test, QQ-plots)
- Coverage testing to validate interval reliability

**Key Tools:** `scikit-learn`, `scipy`, `bootstrap methods`

**Key Question:** Can we trust the prediction margins beyond point estimates?

---

###  Bayesian Linear Regression with Variable Selection
**Goal:** Identify truly important variables while quantifying uncertainty

**Components:**
- Bayesian regression with sparsifying priors (Lasso, Horseshoe)
- MCMC inference (NUTS) for posterior distributions
- Credible intervals for each coefficient
- Comparison with frequentist Lasso

**Key Tools:** `PyMC`, `Stan`, `scikit-learn`

**Key Question:** Which variables truly matter, with quantified uncertainty?

---

###  Hierarchical Bayesian Model for Grouped Data
**Goal:** Model relationships in grouped data with robust group-level estimates

**Components:**
- Mixed-effects models with random intercepts
- Shrinkage estimation for small groups
- Posterior Predictive Checks (PPC) for validation
- Model comparison (WAIC/LOO-CV)

**Key Tools:** `PyMC`, `bambi`, `lme4`

**Key Question:** How do we robustly estimate group-specific effects?

---

### Restaurant Traffic Analysis using SVD
**Goal:** Analyze and predict restaurant attendance patterns

**Components:**
- Matrix construction (Customers × Days, Days × Hours, Dishes × Customers)
- Singular Value Decomposition (SVD) for latent pattern extraction
- Low-rank approximation for pattern discovery
- Traffic prediction and visualization

**Key Tools:** `numpy`, `scipy.linalg`, `matplotlib`, `seaborn`

**Applications:**
- Identify busiest days/hours
- Customer behavior profiling
- Staff scheduling optimization
- Inventory management
- Dish recommendation system

---

##  Technologies Used

### Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

### ML & Data Science
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

### Statistical Analysis & Bayesian Inference
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
`statsmodels` | `PyMC3` | `Stan` | `ArviZ`

### Interpretability & Visualization
`SHAP` | `ELI5` | `matplotlib` | `seaborn` | `plotly`

---

### Installation
```bash
# Clone the repository
git clone https://github.com/vany-gr34/statsitics-and-machine-learning-projects-.git
cd statsitics-and-machine-learning-projects-


```

### Running Projects
Each project is contained in its own Jupyter notebook:
```bash
jupyter notebook
# Navigate to the desired project notebook
```

---

##  Key Concepts Demonstrated

- **Statistical Rigor in ML:** Going beyond accuracy to test hypotheses
- **Uncertainty Quantification:** Confidence/credible intervals for predictions
- **Fairness Testing:** Detecting and quantifying algorithmic bias
- **Bayesian Inference:** Full posterior distributions vs. point estimates
- **Model Validation:** Proper testing beyond training metrics
- **Dimensionality Reduction:** SVD for pattern discovery

---

##  Learning Outcomes

This repository demonstrates proficiency in:
- Combining ML with statistical hypothesis testing
- Implementing Bayesian methods (MCMC, priors, posteriors)
- Performing rigorous model validation and diagnostics
- Quantifying and communicating uncertainty
- Detecting bias and drift in ML systems
- Matrix factorization techniques (SVD)

---

##  Contributing

Contributions are welcome! Feel free to:
- Add new mini-projects
- Improve existing implementations
- Enhance documentation
- Report issues

---

## Academic Context

These projects are part of a statistics and machine learning course focusing on:
- Practical application of statistical inference
- Integration of ML with statistical testing
- Real-world problem solving with quantified uncertainty
- Bayesian and frequentist approaches

---


##  License

This project is open source and available for educational purposes.

---

⭐ **Star this repo** if you find it helpful for learning statistical ML!
