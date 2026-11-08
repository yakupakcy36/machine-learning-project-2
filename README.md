# Fake News Prediction

Small learning project. The notebook `fake news prediction (1).ipynb` trains TF-IDF with logistic regression.

Data: A fake news CSV (train 2.csv), classifying news text as real or fake. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares TF-IDF with logistic regression, complement naive Bayes and passive-aggressive with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (1200 synthetic short texts), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | accuracy | f1 | roc_auc |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8925 | 0.8915 | 0.9075 |
| TF-IDF + Complement Naive Bayes | 0.8775 | 0.877 | 0.9029 |
| TF-IDF + Passive-Aggressive | 0.8217 | 0.8211 | 0.879 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `fake news prediction (1).ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
