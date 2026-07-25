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

# Part 2 – The Stochastic Argument (Theoretical Foundation)

## Q3. Refuting Determinism

Although the laws of physics are deterministic, it is **impossible to derive an exact analytical function** $y=f(x)$ for predicting daily solar energy output in this environment because many influential factors are either **unmeasured**, **unknown**, or **inherently random**.

For example, two days may have the same sunlight duration, yet produce different amounts of energy due to:

- Dust accumulation on the solar panels
- Ambient temperature variations
- Cloud density
- Wind speed
- Sensor measurement errors
- Equipment efficiency degradation

These factors introduce **Aleatoric Uncertainty**, which refers to the natural randomness or noise present in the data generation process. Unlike uncertainty caused by a lack of knowledge (epistemic uncertainty), aleatoric uncertainty **cannot be completely eliminated**, even if we collect more data.

Therefore, instead of assuming a deterministic relationship

$$
y=f(x),
$$

we model the output using a stochastic formulation

$$
y=f(x)+\epsilon,
$$

where:

- $f(x)$ represents the systematic relationship learned by the model.
- $\epsilon$ represents random noise (aleatoric uncertainty), accounting for unpredictable environmental factors and measurement errors.

Consequently, the machine learning model does not attempt to predict the exact energy output. Instead, it learns the underlying trend while treating the residual variation as random noise.

---

# Part 3 – The Optimization Cycle (Procedural Application)

Given:

$$
x^{(1)}=
\begin{bmatrix}
2\\
1
\end{bmatrix},
\qquad
y^{(1)}=5
$$

Current parameters:

$$
w=
\begin{bmatrix}
1.5\\
1.0
\end{bmatrix},
\qquad
b=0
$$

---

## Q4. Manual Forward Pass

The prediction of a linear model is

$$
\hat{y}=w^Tx+b.
$$

Substitute the given values:

$$
\hat{y}
=
\begin{bmatrix}
1.5 & 1.0
\end{bmatrix}
\begin{bmatrix}
2\\
1
\end{bmatrix}
+0
$$

Perform the matrix multiplication:

$$
= (1.5)(2) + (1.0)(1)
$$

$$
=3+1
$$

$$
=4
$$

Therefore,

$$
\boxed{\hat{y}^{(1)}=4}
$$

---

## Q5. Loss Calculation

The Squared Error (SE) for a single sample is

$$
SE=(\hat{y}-y)^2.
$$

Substitute the values:

$$
SE=(4-5)^2
$$

$$
=(-1)^2
$$

$$
=1
$$

Therefore,

$$
\boxed{SE=1}
$$

### Interpretation

The squared error measures how far the model's prediction is from the actual value.

- If the loss is **0**, the prediction is perfect.
- A larger loss indicates a greater prediction error.

For this sample, the loss of **1** tells the optimizer that the model **underestimated** the true energy output by **1 MW**. The optimizer uses this scalar value, together with the gradient, to determine how the model parameters should be adjusted to reduce future prediction errors.

---

## Q6. Gradient Intuition

The optimizer aims to minimize the loss by updating the parameters in the direction that reduces prediction error.

Suppose the prediction is **too high** (overestimation), meaning

$$
\hat{y}>y.
$$

To reduce the prediction, the optimizer should:

- Decrease the weights that contribute positively to the prediction.
- Decrease the bias if it also contributes to the overestimation.

Conceptually, the parameter update is

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\alpha\nabla J(\theta),
$$

where:

- $\alpha$ is the learning rate.
- $\nabla J(\theta)$ is the gradient of the loss function.

The optimizer repeatedly moves the parameters downhill on the loss surface until reaching a minimum.

### Simple Loss Surface Illustration

```text
Loss J(θ)
 ^
 |                    ● Initial Parameters
 |                  /
 |                /
 |              /
 |            /
 |          /
 |        ● Minimum Loss
 +---------------------------------> Parameters θ
```

The optimizer follows the negative gradient (downhill direction), gradually updating the parameters so that the predicted energy output becomes closer to the actual value and the loss decreases over successive iterations.
