## Problem setting
In the nonparametric setting, we need to fit an curve, we suppose the data sampling from the mode $Y = m(X) + \epsilon$, where $X$ is predictors of dimension $d$, $\epsilon$ is a random error satisfying $E(\epsilon|X) = 0$ almost surely,  deep feed neural network is greately useful to fit the unkown function $m$, while we focus on the statistical non-parametric method to estimate $m$ since there are  rich and complete theoretical results. However the mean squared of any nonparametric estimator of a smooth (twice differentiable) curve will typically have mean squared error of the form  
$$MSE \approx  \frac{c}{n^{4/(4+d)}}$$
for some $c>0$
If we want the mse to be equal to some small number δ, we can set $MSE = \delta$ and solve for n. We find that
$$n \sim (\frac{c}{\delta})^{d/4}$$
which grows exponentially with dimension d. Which called curse of dimensionality. The reason for this phenomenon is that smoothing involves estimating a function $m(x)$ using data points in a local neighborhood of $x$. But in a high dimensional problem, the data are very sparse, so local neighborhoods contain very few points.
                                
We want to learn an orientation $\theta$  adapts to the different data set, then use the length of  data projected onto this orientation to create nonparametric estimator, which escape the problem of curse of dimensionality. 

## Procedure

$$\min \limits_{} \sum \limits_{j = 1}^{n} \sum \limits_{i = 1}^{n}\left[ y_i - a_j -  b_j \beta^T(z_i - z_j)\right]^2 \omega_{ij} + \lambda \sum \limits_{j=1}^{n} \left|b_j\right| \sum \limits_{k=1}^{p}\beta_k,
$$
