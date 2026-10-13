# Information Retention and Scaling in Time-Series Transformers

**A computational study of how stochastic dynamics, observation structure, and input corruption affect the information recovered by a masked time-series Transformer.**

[![Python](https://img.shields.io/badge/Python-Research_Code-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Core_Models-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Notebooks](https://img.shields.io/badge/Workflow-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

This repository develops a synthetic experimental framework for studying **information retention under progressively degraded observations**. Binary stochastic processes are aggregated into overlapping node signals, reconstructed by a Transformer, and evaluated using masked reconstruction error, Shannon entropy, and mutual information.

The scaling experiments vary the number of observed signals, $K$, and measure how retention curves and operational failure thresholds change with corruption strength. Related notebooks explore random-walk drift, mean-reverting drift, context-window length, and several corruption mechanisms.

Developed by **[Ali Khabazian](https://github.com/Alikhabazian0)**.

## Contents

- [Research questions](#research-questions)
- [Experimental framework](#experimental-framework)
- [Information-theoretic evaluation](#information-theoretic-evaluation)
- [Corruption and drift experiments](#corruption-and-drift-experiments)
- [Scaling with the number of observed signals](#scaling-with-the-number-of-observed-signals)
- [Configuration](#configuration)
- [Repository guide](#repository-guide)
- [Running the experiments](#running-the-experiments)
- [Supplementary forecasting pipelines](#supplementary-forecasting-pipelines)
- [Interpretation and reproducibility](#interpretation-and-reproducibility)
- [References](#references)

## Research questions

1. How much information about clean node signals can a trained model recover from incomplete or corrupted context?
2. How do changes in the underlying stochastic drift affect reconstruction and information retention?
3. How do local missing values, missing nodes, and missing temporal blocks differ in their effects?
4. Does changing $K$ alter the location or width of the decline in retention?
5. How does a Transformer compare with simple mean-based reconstruction baselines?

The central task is **masked reconstruction within a temporal window**. The core Transformer uses a bidirectional encoder, so these experiments evaluate contextual recovery rather than a strictly causal forecasting task. Separate notebooks provide forecasting pipelines.

## Experimental framework

### 1. Latent binary dynamics

The synthetic world is a binary state matrix:

$$
X\in\{0,1\}^{T\times D},
$$

where $T$ is the number of time steps and $D$ is the number of latent coordinates. For coordinate $d$, the base transition logits are sampled as:

$$
\alpha_d,\beta_d\sim\mathcal{N}(\mu,\sigma^2).
$$

Time-varying transition probabilities depend on a drift variable $\epsilon_{t,d}$:

$$
p_{t,d}=\frac{1}{1+\exp[-(\alpha_d+\epsilon_{t,d})]},
\qquad
q_{t,d}=\frac{1}{1+\exp[-(\beta_d+\epsilon_{t,d})]}.
$$

Here $p_{t,d}$ is the probability of a transition from zero to one, and $q_{t,d}$ is the probability of a transition from one to zero:

$$
X_{t,d}\sim
\begin{cases}
\mathrm{Bernoulli}(p_{t,d}),&X_{t-1,d}=0,\\
\mathrm{Bernoulli}(1-q_{t,d}),&X_{t-1,d}=1.
\end{cases}
$$

The initial coordinates are sampled from a Bernoulli distribution with probability $0.5$.

### 2. Random-walk and mean-reverting drift

The main simulation notebooks include a random-walk drift:

$$
\epsilon_{t,d}=\epsilon_{t-1,d}+\eta_{t,d},
\qquad
\eta_{t,d}\sim\mathcal{N}(0,\sigma_{\mathrm{drift}}^2).
$$

The Ornstein-Uhlenbeck-inspired notebooks use the discrete mean-reverting update:

$$
\epsilon_{t,d}=(1-\lambda)\epsilon_{t-1,d}
+\lambda\epsilon_0+\eta_{t,d}.
$$

The parameter $\lambda$ controls mean reversion, while $\sigma_{\mathrm{drift}}$ controls the innovation scale. This is the update implemented in the notebooks; no separate continuous-time step or exact continuous-time OU transition is specified. Mean reversion acts on the latent drift, while the observed state remains binary.

### 3. Overlapping observations

Each observed signal averages a subset $S_k$ of latent coordinates:

$$
Z_{t,k}=\frac{1}{|S_k|}\sum_{d\in S_k}X_{t,d},
\qquad
Z\in[0,1]^{T\times K}.
$$

Overlapping subsets create shared information among the observed signals. In the default scaling configuration, each subset contains 25 coordinates. The construction explicitly selects five coordinates from an earlier subset, then samples the remaining coordinates. Additional overlap can occur, so `overlap_target=5` is an overlap-construction parameter rather than a guarantee of exactly five shared coordinates for every pair.

The main notebook also constructs thresholded correlation networks and calculates graph statistics such as connected-component sizes, degree, density, and algebraic connectivity. These describe the observation structure; the core reconstruction model is a temporal Transformer rather than a graph neural network.

### 4. Masked reconstruction

A window of length $L$ supplies the model with $K$ observed signals. Randomly selected entries are set to zero, and the original values provide reconstruction targets. The training loss is restricted to the masked entries:

$$
\mathcal{L}_{\mathrm{masked}}
=\frac{\sum_{b,t,k}m_{b,t,k}
(\widehat Z_{b,t,k}-Z_{b,t,k})^2}
{\sum_{b,t,k}m_{b,t,k}},
$$

where $m_{b,t,k}=1$ identifies a masked location. The implementation adds a small numerical guard to the denominator.

The scaling model contains a linear projection from $K$ input channels, learned positional embeddings, stacked Transformer encoder layers, and a linear projection back to $K$ channels. Its feed-forward blocks use GELU activation. Training uses AdamW.

The random reconstruction mask is part of the learning objective. The additional corruption sweep is applied to the observed input signals during evaluation, with clean targets retained by `PairedMaskedWindowDataset`.

## Information-theoretic evaluation

Predictions and targets at masked positions are pooled and discretized before computing empirical information metrics. The scaling configuration uses `np.linspace(0, 1, 21)` as its bin-edge array.

| Quantity | Definition | Interpretation |
|---|---|---|
| Masked MSE | Mean squared error on evaluated masked entries | Reconstruction accuracy |
| Target entropy | $H(Z)=-\sum_z p(z)\log_2 p(z)$ | Uncertainty in the discretized target values |
| Mutual information | $I(Z;\widehat Z)=H(Z)+H(\widehat Z)-H(Z,\widehat Z)$ | Statistical information shared by predictions and targets |
| Retention | $R=I(Z;\widehat Z)/H(Z)$ | Shared information normalized by target entropy |

The main simulation notebook also contains joint-entropy and conditional-entropy helpers. All histogram-based entropy calculations use base-two logarithms, so entropy and mutual information are measured in **bits**. Retention is dimensionless; the code returns zero when target entropy is too small for a stable normalization.

These quantities describe empirical distributions of binned scalar reconstruction pairs. They are not estimates of the full sequence entropy rate. Predictions outside the nominal interval are assigned categories by `np.digitize`; the discretization function does not clip their values.

### Baselines

The core comparisons include a **window-node-mean baseline**, which predicts each node using its mean over the supplied window. The main simulation notebook also includes a global training-node-mean baseline.

The window baseline averages the zero-filled input, including masked or corrupted zeros. It therefore reflects the implemented reference method, rather than a mean computed exclusively from valid observations.

## Corruption and drift experiments

The notebooks separate the underlying stochastic dynamics from the mechanisms used to degrade observations.

| Mechanism | Implementation | Aspect examined |
|---|---|---|
| Bit flipping | Flip latent binary entries with probability $\delta$ | Perturbations to the underlying state |
| Element dropout | Set individual observed entries to zero with probability $\delta$ | Scattered missing observations |
| Node dropout | Set selected signal columns to zero across all times | Loss of whole observed channels |
| Block dropout | Set selected contiguous time blocks to zero across all channels | Temporally concentrated missing information |
| Partial time shuffling | Permute a selected fraction of time indices | Disruption of temporal ordering |
| Drift sweep | Regenerate trajectories at different $\sigma_{\mathrm{drift}}$ | Changes in the dynamics seen by a fixed trained model |

For node and block dropout, $\delta$ is a selection probability; the realized fraction of removed data depends on the random draw. Corruption seeds are held fixed in the principal sweeps, allowing controlled comparisons over corruption strengths.

Related notebooks examine different context windows, including inspected variants with 16, 64, and 128 time steps. Several OU drift sweeps use:

```python
sigmas = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
```

## Scaling with the number of observed signals

The principal scaling notebooks vary:

$$
K\in\{20,40,80,160\}.
$$

The latent dimension, generating process, subset size, and internal Transformer dimensions remain fixed in the default configuration. A separate model is trained for each $K$ and evaluated over the corruption grid.

Here **$K$ means the number of observed channels**. It does not denote the Transformer embedding width. Changing $K$ also changes the input/output projection sizes and the observation redundancy, so this is an observation-size experiment with associated model-parameter changes.

### Operational thresholds

For each retention curve, let $R_0=R(\delta=0)$. The scaling helper records the first sampled corruption level where retention falls below a fraction of its clean value:

$$
\delta_a=\min\{\delta_j:R(\delta_j)\le aR_0\},
\qquad a\in\{0.9,0.5,0.1\}.
$$

Its transition-width diagnostic is:

$$
w(K)=\delta_{0.1}(K)-\delta_{0.9}(K).
$$

The scaling helper uses grid crossings without interpolation and returns `None` for thresholds not reached. A width is therefore available only when both crossings exist.

The main simulation notebook additionally explores interpolated loss-of-baseline-advantage thresholds, absolute retention floors, and a knee heuristic based on second differences. These are operational diagnostics with different definitions, not interchangeable estimates of a universal critical point.

### Committed example figures

![Retention curves for different numbers of observed channels](Resutls/ks%20scaling/Retention%20vs%20%CE%B4%20for%20K%3D160.png)

The committed figure shows retention declining with increasing block-dropout probability for all four displayed values of $K$. The ordering of the curves also illustrates that increasing the number of observed channels does not automatically increase retention under this configuration. Despite the image filename, the plot contains all four $K$ values.

![Operational transition-width diagnostic](Resutls/ks%20scaling/Knee%20width%20vs%20K%20%28transition%20sharpening%20test%29.png)

The committed width plot contains a single plotted point near $K=160$, with width approximately $0.40$. It does not by itself establish size-dependent narrowing or a finite-size scaling law. The repository provides the experimental machinery for examining those questions; conclusions require valid thresholds across sizes and repeated runs.

## Configuration

The table below describes the `CFG` dictionary in the principal scaling notebooks. Other notebooks contain their own configurations.

| Group | Parameter | Default |
|---|---|---|
| Synthetic world | Time steps / latent coordinates | $T=20{,}000$, $D=256$ |
| Transition logits | Mean / standard deviation | $\mu=-1.5$, $\sigma=0.3$ |
| OU-inspired drift | Mean reversion / innovation scale / long-run mean | $\lambda=0.03$, $\sigma_{\mathrm{drift}}=0.001$, $\epsilon_0=0$ |
| Observations | Channel counts | `20, 40, 80, 160` |
| Subsets | Size / explicit overlap component | `25` / `5` |
| Temporal window | Length | `128` |
| Reconstruction mask | Probability | `0.15` |
| Temporal split | Training fraction | `0.8` |
| Training | Epochs / batch size / learning rate | `5` / `64` / `3e-4` |
| Transformer | Embedding width / attention heads / encoder layers | `128` / `4` / `3` |
| Feed-forward block | Hidden width / dropout | `256` / `0.1` |
| Evaluation corruption | Mode / block length | `block` / `64` |
| Corruption sweep | Probability grid | 21 values from `0.0` to `0.5` |
| Information metrics | Bin edges | `np.linspace(0, 1, 21)` |
| NumPy seeds | World / subsets / corruption | `123` / `123` / `999` |
| Output location | Scaling results | `k_scaling_results/` |

## Repository guide

| Location | Contents |
|---|---|
| [`ks_scaling_src/`](ks_scaling_src/) | Main channel-count scaling experiments, retention sweeps, and threshold analysis. |
| [`main_src/Simulation - Main.ipynb`](main_src/Simulation%20-%20Main.ipynb) | Binary dynamics, observation construction, correlation networks, masked reconstruction, information metrics, and corruption diagnostics. |
| [`main_src/`](main_src/) | OU/drift variants, window-length comparisons, corruption-mode experiments, and saved-model evaluation notebooks. |
| [`data_src/`](data_src/) | Dataset preparation, scaling, walk-forward forecasting, and classical/deep-learning model pipelines. |
| [`Raw Data/`](Raw%20Data/) | Committed process illustrations; forecasting notebooks refer to additional local CSV inputs. |
| [`Resutls/main/`](Resutls/main/) | Figures from the main experiments. |
| [`Resutls/ks scaling/`](Resutls/ks%20scaling/) | Committed scaling and retention figures. |

The directory name `Resutls` is retained here to match the repository. Individual notebooks preserve exploratory variants; later definitions within a notebook can replace earlier definitions of the same function.

## Running the experiments

### 1. Install the core notebook environment

```bash
git clone https://github.com/Alikhabazian0/KS-Scaling.git information-retention-scaling
cd information-retention-scaling
python -m pip install numpy matplotlib torch jupyterlab
```

For a CUDA-enabled installation, use the command appropriate to your system from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). The notebooks select CUDA when available and otherwise use the CPU.

### 2. Start with the principal scaling notebook

```bash
jupyter lab "ks_scaling_src/K_scaling_experiment.ipynb"
```

Run cells in order from a fresh kernel. Review the `CFG` dictionary before the training loop. The workflow generates a latent world, builds observations for each $K$, trains on clean masked signals, evaluates corrupted inputs against clean targets, and saves metrics and model weights.

For an initial functional check, reduce the workload before executing the data-generation and training cells:

```python
CFG.update(
    T=3000,
    K_list=[20],
    epochs=1,
    deltas=np.linspace(0.0, 0.5, 5),
)
```

This smaller configuration checks the workflow; use the full settings for comparisons with the committed experiment configuration. If inputs have already been generated, restart the relevant cells so that the changed configuration takes effect.

### 3. Inspect saved outputs

The scaling loop writes model weights (`.pt`), training histories (`.json`), and per-$K$ result files (`.json`) under its configured output directory. Result records include $K$, corruption strength, target entropy, mutual information, and normalized retention for the Transformer and window baseline.

Some downstream notebook cells load existing files and plot them. Generate those files first, or update the paths to an existing run. The supplementary forecasting notebooks also contain machine-specific Windows paths that must be changed for a new environment.

## Supplementary forecasting pipelines

The `data_src/` notebooks extend the project with sequence preparation and forecasting experiments on local process CSV files:

- Lag features and sequence windows, with scaling fitted on training windows.
- Expanding or rolling walk-forward splits.
- Linear regression, random forest, and multilayer-perceptron baselines.
- ARIMA/SARIMA/GARCH selection helpers.
- TensorFlow/Keras LSTM, GRU, TCN, and causal-attention Transformer builders.
- Metrics including MSE, MAE, SMAPE, and directional accuracy, with predictions and training metadata saved by the pipelines.

These notebooks use a separate dependency stack, including `pandas`, `scikit-learn`, `joblib`, `statsmodels`, `pmdarima`, `arch`, and `tensorflow`. Inspect the imports of the selected notebook for additional experimental dependencies. Their forecasting objectives and data inputs are distinct from the core PyTorch masked-reconstruction experiment.

## Interpretation and reproducibility

The principal comparisons hold the latent world and corruption draws fixed across configured sweeps, which helps isolate changes in observed-channel count and corruption strength. The following implementation details matter when interpreting results:

- **Initialization:** NumPy seeds are specified, but the principal scaling notebook does not explicitly seed PyTorch model initialization. Exact training reproduction requires recording or setting the relevant PyTorch and device settings.
- **Validation context:** validation data begin at `split - window`. Masks can cover entries in that overlapping context, so this is the implemented windowed reconstruction evaluation rather than a fully disjoint target evaluation.
- **Pooling:** information estimates pool masked scalar entries across windows and channels. Their empirical distribution depends on the discretization, overlapping observations, and evaluation sample.
- **Baseline definition:** the window mean uses zero-filled inputs; alternative treatment of missing values would change the comparison.
- **Scaling variable:** increasing $K$ changes both observation structure and projection parameter counts while leaving internal embedding width fixed.
- **Threshold resolution:** crossings depend on the sampled corruption grid. Missing crossings produce missing widths rather than evidence that the width is zero.
- **Research claims:** the notebooks investigate retention degradation and possible transition sharpening. A sharp-looking curve or one width measurement alone does not demonstrate a thermodynamic phase transition or a universal scaling exponent.

For reproducible runs, preserve the notebook version or Git commit, complete configuration, environment versions, seeds, training history, model weights, raw sweep metrics, and threshold definitions. Compare repeated realizations before assigning uncertainty intervals or generalizing the observed curve ordering.

## References

The experiment design and notebook implementation are the work of **Ali Khabazian**. The following are foundational references for the information-theoretic, attention-based, and stochastic-process concepts used here:

1. **Shannon, C. E. (1948).** A mathematical theory of communication. *The Bell System Technical Journal, 27*(3), 379-423. [DOI: 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x).
2. **Vaswani, A., et al. (2017).** Attention Is All You Need. *Advances in Neural Information Processing Systems, 30*. [Paper](https://arxiv.org/abs/1706.03762).
3. **Uhlenbeck, G. E., & Ornstein, L. S. (1930).** On the theory of the Brownian motion. *Physical Review, 36*, 823-841. [DOI: 10.1103/PhysRev.36.823](https://doi.org/10.1103/PhysRev.36.823).

When using this implementation, acknowledge the repository and cite the specific commit used. Cite the foundational literature as appropriate to the methods used in your analysis.
