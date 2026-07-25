## **Q1. Data Vectorization**

### **1. Mathematical Definition of Dataset (D)**

Suppose we collect data for **(N = 365)** days.

The dataset can be represented as

[
D={(x^{(i)},y^{(i)})}_{i=1}^{365}
]

where:

* (x^{(i)}) = feature vector for day (i)
* (y^{(i)}) = actual solar energy produced on day (i) (MW)

---

### **2. Feature Vector**

Instead of only using sunlight duration, we include additional environmental variables that affect solar energy production.

One possible feature vector is

[
x^{(i)}=
\begin{bmatrix}
1\
x_1^{(i)}\
x_2^{(i)}\
x_3^{(i)}
\end{bmatrix}
]

where

* **1** = bias term
* (x_1) = Sunlight duration (hours)
* (x_2) = Ambient temperature (°C)
* (x_3) = Dust accumulation index on solar panels

Additional useful features could also include:

* Cloud density
* Solar irradiance
* Humidity
* Wind speed

The corresponding label is

[
y^{(i)}=\text{Energy Output (MW)}
]

---

### **3. Feature Matrix**

For 365 days and 3 environmental features plus one bias term,

[
X=
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & x_3^{(1)}\
1 & x_1^{(2)} & x_2^{(2)} & x_3^{(2)}\
\vdots & \vdots & \vdots & \vdots\
1 & x_1^{(365)} & x_2^{(365)} & x_3^{(365)}
\end{bmatrix}
]

Dimensions:

[
X\in\mathbb{R}^{365\times4}
]

because there are

* 365 observations
* 4 columns (bias + 3 features)

---

### **4. Label Vector**

[
Y=
\begin{bmatrix}
y^{(1)}\
y^{(2)}\
\vdots\
y^{(365)}
\end{bmatrix}
]

Dimensions:

[
Y\in\mathbb{R}^{365\times1}
]

---

## **Q2. The Hypothesis Space**

A suitable **linear hypothesis** for predicting solar energy output is

[
h_\theta(x)=\theta^Tx
]

or equivalently

[
h_\theta(x)
===========

\theta_0
+\theta_1x_1
+\theta_2x_2
+\theta_3x_3
]

where

* (x_1) = sunlight duration
* (x_2) = ambient temperature
* (x_3) = dust accumulation

---

### Interpretation of the Parameter Vector

The parameter vector is

[
\theta=
\begin{bmatrix}
\theta_0\
\theta_1\
\theta_2\
\theta_3
\end{bmatrix}
]

where:

* **(\theta_0)** : Bias/intercept, representing the baseline predicted energy output when all features are zero.
* **(\theta_1)** : Effect of sunlight duration on energy production.
* **(\theta_2)** : Effect of ambient temperature on energy production.
* **(\theta_3)** : Effect of dust accumulation on energy production.

Each parameter measures how much the predicted energy output changes when its corresponding feature increases by one unit, assuming the other features remain constant.

---

### Meaning of a High Positive Weight (w_j)

If a feature has a **large positive weight**,

[
w_j \gg 0,
]

then that feature has a **strong positive influence** on the predicted solar energy output.

For example:

* Large positive **(w_1)**:

  * Longer sunlight duration significantly increases energy production.
* Large positive **(w_2)**:

  * Higher temperatures are associated with increased energy output (within the observed data range).
* Large positive **(w_3)**:

  * Greater dust accumulation would increase energy production, which is generally unrealistic. In practice, we would expect the dust coefficient to be **negative**, since dust blocks sunlight and reduces panel efficiency.

Thus, **the sign** of a weight indicates the direction of the relationship (positive or negative), while **its magnitude** indicates how strongly that feature influences the prediction.
