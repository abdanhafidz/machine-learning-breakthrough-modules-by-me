# Part 1 – Abstraction and Formal Representation

## Q1. Data Vectorization

### 1. Mathematical Definition of the Dataset

Suppose we collect data for **$N = 365$** days.

The dataset is defined as

$$
D=\{(x^{(i)},y^{(i)})\}_{i=1}^{365}
$$

where:

- $x^{(i)}$ is the feature vector for day $i$.
- $y^{(i)}$ is the actual solar energy output (MW) for day $i$.

---

### 2. Feature Vector

To better capture the factors affecting solar energy production, we extract multiple environmental features instead of relying solely on sunlight duration.

A possible feature vector is

$$
x^{(i)}=
\begin{bmatrix}
1\\
x_1^{(i)}\\
x_2^{(i)}\\
x_3^{(i)}
\end{bmatrix}
$$

where:

- $1$ = Bias term
- $x_1$ = Sunlight duration (hours)
- $x_2$ = Ambient temperature (°C)
- $x_3$ = Dust accumulation index on the solar panels

Additional features that could further improve prediction include:

- Cloud density
- Solar irradiance
- Humidity
- Wind speed

The corresponding label is

$$
y^{(i)}=\text{Energy Output (MW)}
$$

---

### 3. Feature Matrix

For 365 observations, the feature matrix becomes

$$
X=
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & x_3^{(1)}\\
1 & x_1^{(2)} & x_2^{(2)} & x_3^{(2)}\\
\vdots & \vdots & \vdots & \vdots\\
1 & x_1^{(365)} & x_2^{(365)} & x_3^{(365)}
\end{bmatrix}
$$

Its dimensions are

$$
X\in\mathbb{R}^{365\times4}
$$

because there are:

- **365 rows** (days)
- **4 columns** (1 bias term + 3 features)

---

### 4. Label Vector

The target vector is

$$
Y=
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(365)}
\end{bmatrix}
$$

with dimensions

$$
Y\in\mathbb{R}^{365\times1}
$$

---

# Q2. The Hypothesis Space

A suitable **linear hypothesis model** is

$$
h_\theta(x)=\theta^Tx
$$

or equivalently,

$$
h_\theta(x)
=
\theta_0
+\theta_1x_1
+\theta_2x_2
+\theta_3x_3
$$

where:

- $x_1$ = Sunlight duration
- $x_2$ = Ambient temperature
- $x_3$ = Dust accumulation index

---

## Interpretation of the Parameter Vector

The parameter vector is

$$
\theta=
\begin{bmatrix}
\theta_0\\
\theta_1\\
\theta_2\\
\theta_3
\end{bmatrix}
$$

where:

- **$\theta_0$** is the bias (intercept), representing the baseline predicted energy output when all features are zero.
- **$\theta_1$** measures the influence of sunlight duration on energy production.
- **$\theta_2$** measures the influence of ambient temperature on energy production.
- **$\theta_3$** measures the influence of dust accumulation on energy production.

Each parameter represents the expected change in predicted energy output for a one-unit increase in its corresponding feature while keeping the other features constant.

---

## Interpretation of a High Positive Weight

If a feature has a **large positive weight**,

$$
w_j \gg 0,
$$

then that feature has a **strong positive influence** on the predicted solar energy output.

For example:

- A large positive **$w_1$** indicates that increasing sunlight duration significantly increases energy production.
- A large positive **$w_2$** suggests that higher ambient temperatures are associated with greater energy output (within the observed data range).
- A large positive **$w_3$** would imply that increased dust accumulation raises energy production, which is generally unrealistic. In practice, the coefficient for dust accumulation is expected to be **negative**, since dust reduces the efficiency of solar panels.

Therefore:

- A **positive weight** indicates a positive relationship between the feature and energy output.
- A **negative weight** indicates an inverse relationship.
- The **magnitude** of the weight indicates how strongly the feature influences the prediction.
