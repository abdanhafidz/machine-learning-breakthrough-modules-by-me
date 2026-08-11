# Feature Scaling

Feature scaling is used to make numerical features have comparable scales.

This is especially useful for optimization algorithms such as Gradient Descent.

---

## 1. Standardization — Standard Scaler

Standardization transforms the feature so that its mean becomes approximately $0$ and its standard deviation becomes approximately $1$.

$$
x' = \frac{x-\mu}{\sigma}
$$

Where:

- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

### NumPy Implementation

```python
mean = X.mean(axis=0)
std = X.std(axis=0)

X_scaled = (X - mean) / std
```

---

## 2. Min-Max Scaling

Min-Max Scaling transforms the feature into the range $[0,1]$.

$$
x' =
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
$$

### NumPy Implementation

```python
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (X - X_min) / (X_max - X_min)
```

---

## 3. Min-Max Scaling to $[-1,1]$

This method transforms the feature into the range $[-1,1]$.

$$
x' =
2\left(
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
\right)-1
$$

### NumPy Implementation

```python
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (
    2 * (X - X_min) / (X_max - X_min)
) - 1
```

---

## 4. Mean Normalization

Mean normalization centers the feature around $0$ while considering its range.

$$
x' =
\frac{x-\mu}
{x_{\max}-x_{\min}}
$$

### NumPy Implementation

```python
mean = X.mean(axis=0)
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (X - mean) / (X_max - X_min)
```

---

## 5. Max Absolute Scaling

Each feature is divided by its maximum absolute value.

$$
x' =
\frac{x}
{\max(|x|)}
$$

The resulting values are generally in the range $[-1,1]$.

### NumPy Implementation

```python
max_abs = np.abs(X).max(axis=0)

X_scaled = X / max_abs
```

---

## 6. Robust Scaling

Robust Scaling uses the median and Interquartile Range (IQR).

$$
x' =
\frac{x-\operatorname{median}(x)}
{Q_3-Q_1}
$$

Where:

- $Q_1$ = first quartile
- $Q_3$ = third quartile
- $Q_3-Q_1$ = Interquartile Range (IQR)

This method is less sensitive to outliers.

### NumPy Implementation

```python
median = np.median(X, axis=0)

Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)

IQR = Q3 - Q1

X_scaled = (X - median) / IQR
```

---

## Comparison

| Method | Formula | Typical Range | Gradient Descent |
|---|---|---|---|
| Standard Scaler | $\frac{x-\mu}{\sigma}$ | Unbounded | ⭐⭐⭐⭐⭐ |
| Min-Max | $\frac{x-x_{\min}}{x_{\max}-x_{\min}}$ | $[0,1]$ | ⭐⭐⭐⭐ |
| Min-Max $[-1,1]$ | $2\frac{x-x_{\min}}{x_{\max}-x_{\min}}-1$ | $[-1,1]$ | ⭐⭐⭐⭐ |
| Robust Scaler | $\frac{x-\operatorname{median}(x)}{Q_3-Q_1}$ | Unbounded | ⭐⭐⭐⭐ |
| Mean Normalization | $\frac{x-\mu}{x_{\max}-x_{\min}}$ | Not fixed | ⭐⭐⭐ |
| Max Absolute | $\frac{x}{\max(|x|)}$ | $[-1,1]$ | ⭐⭐⭐ |

---

## Important: Train-Test Scaling

When working with training and test data, calculate the scaling parameters **only from the training data**.

For Standard Scaler:

$$
\mu_{\text{train}}
=
\operatorname{mean}(X_{\text{train}})
$$

$$
\sigma_{\text{train}}
=
\operatorname{std}(X_{\text{train}})
$$

Then use these same parameters for both training and test data.

### Training Data

$$
X'_{\text{train}}
=
\frac{X_{\text{train}}-\mu_{\text{train}}}
{\sigma_{\text{train}}}
$$

### Test Data

$$
X'_{\text{test}}
=
\frac{X_{\text{test}}-\mu_{\text{train}}}
{\sigma_{\text{train}}}
$$

> **Do not calculate a separate mean and standard deviation from the test set.**
