

### 1. Standardization — Standard Scaler

Rumus per fitur:

[
x' = \frac{x-\mu}{\sigma}
]

```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

Karakteristik:

* Mean ≈ `0`
* Standard deviation ≈ `1`
* Tidak membatasi range ke `0–1`
* Cocok ketika data punya skala berbeda-beda
* **Sangat umum untuk Gradient Descent**

Contoh:

```text
Original       Standardized
10             -1.34
20             -0.67
30              0.00
40              0.67
50              1.34
```

---

### 2. Min-Max Scaling — MinMax Scaler

Rumus:

[
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
]

```python
X_scaled = (X - X.min(axis=0)) / (
    X.max(axis=0) - X.min(axis=0)
)
```

Hasilnya:

```text
minimum → 0
maximum → 1
```

Contoh:

```text
Original       Min-Max
10             0.00
20             0.25
30             0.50
40             0.75
50             1.00
```

Cocok kalau kamu memang ingin semua fitur berada pada range tertentu.

---

### 3. Min-Max ke `[-1, 1]`

Bisa juga:

[
x' = 2\frac{x-x_{min}}{x_{max}-x_{min}}-1
]

```python
X_scaled = (
    2 * (X - X.min(axis=0))
    / (X.max(axis=0) - X.min(axis=0))
) - 1
```

Hasil:

```text
min → -1
max →  1
```

---

### 4. Mean Normalization

[
x' = \frac{x-\mu}{x_{max}-x_{min}}
]

```python
X_scaled = (
    X - X.mean(axis=0)
) / (
    X.max(axis=0) - X.min(axis=0)
)
```

Ini membuat data berpusat di sekitar `0`, tetapi **tidak menjamin std = 1**.

---

### 5. Max Absolute Scaling

[
x' = \frac{x}{\max(|x|)}
]

```python
X_scaled = X / np.abs(X).max(axis=0)
```

Range umumnya:

```text
[-1, 1]
```

Bagus ketika data sparse atau ingin mempertahankan nilai `0`.

---

### 6. Robust Scaling

Menggunakan median dan IQR:

[
x' = \frac{x-\text{median}(x)}{Q_3-Q_1}
]

Pure NumPy:

```python
median = np.median(X, axis=0)
q1 = np.percentile(X, 25, axis=0)
q3 = np.percentile(X, 75, axis=0)

X_scaled = (X - median) / (q3 - q1)
```

Lebih tahan terhadap **outlier** dibanding Standard Scaler.

---

