<!-- <h1 align="center">
<img src="https://gitlab.com/anna.giabelli/TaxoSS/-/blob/master/img/logo.svg" alt="TaxoSS" width="400">
</h1> -->
<h1 align="center">vec2best: A Unified Framework for Intrinsic Evaluation of Word-Embedding Algorithms</h1>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a>

</p>

---

## Description

vec2best is a library for Python which represents a framework for evaluating word embeddings trained using various methods and hyper-parameters on a range of tasks from the literature. The tool yields a holistic evaluation metric for each model called the $PCE$ (Principal Component Evaluation).

vec2best implements the state-of-the-art intrinsic evaluations tasks of word similarity, word analogy, concept categorisation, and outlier detection over the benchmarks in the following table.

| Task              | Evaluation           | Metric            | Benchmark   |
|-------------------|----------------------|-------------------|-------------|
| Similarity        | Spearman correlation | Cosine similarity | WS353, RG65, RW, MEN, MTurk287, SimLex999, MC30, MTurk771, YP130, Verb143, SimVerb3500, SemEval17, WS353REL, WS353SIM    |
| Analogy           | Accuracy             | 3CosAdd, 3CosMul           | Google, MSR         |
|                   | Spearman correlation | 3CosAdd           | SemEval2012 |
| Categorization    | Purity               | Clustering        | AP, BLESS, BM (battig), ESSLI 1a, ESSLI 2b, ESSLI 2c    |
| Outlier detection | Accuracy             | Compactness score | 8-8-8, WordSim500  |


## Requirements

- Python 3.6
- scikit-learn
- six
- word-embeddings-benchmarks

The package also relies on a modified version on the following repositories for outlier detection:
- https://github.com/peblair/wiki-sem-500
- http://lcl.uniroma1.it/outlier-detection/

## Installation

vec2best can be installed through `pip` (the Python package manager) in the following way:

```bash
pip install vec2best
```

## Usage

To compute the $PCE$ you need to apply the function `compute_pce(path_to_model)` and the only parameter that you need to set is the path in which you saved the embedding models (in a .vec or .txt format) you want to evaluate.

The function `compute_pce(path_to_model)` has other six parameters `(categorization=True, similarity=True}, analogy=True, outlier_detection=True, pce_min=True, pce_max=True, pce_mean=True)`  set by default as `True`, and so the output consists in the evaluation of the models over the three tasks and over the $PCE^{MIN}$, $PCE^{MAX}$, $PCE^{MEAN}$. By setting some of those parameters as `False`, the $PCE$ can be computed over a subset of those tasks or the evaluation could be computed only for one or two of the three types of $PCE$.

The output is saved in the folder _results/pce_, and the output on the screen shows the percentage of explained variance of the first principal component, and the top 3 models according to the chosen $PCE$. 

See the following example:

```python
from vec2best import compute_pce
path_to_model = 'data/example_models' 
compute_pce(path_to_model, analogy=False,outlier_detection=False, 
pce_max=False, pce_mean=False)
```

The output will look like:

```python
PCE min - percentage of explained variance: 0.95
                                 categorization    similarity    PCE_min
example_models/ft_0_5_50_5.vec   0.38              0.29          1.00
example_models/glove_5_50_5.vec  0.41              0.25          0.94
example_models/wv2_model_11.vec  0.24              0.17          0.34
```