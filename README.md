# SIM_NNG
The non-negative garrotte method in single-index model



$\begin{equation}\label{equ:2.5}	
\min \limits_{\mbox{\tiny $\begin{array}{c}
		\beta_k > 0, k =1,\dots,p \\
		\beta^T(\mbox{diag}(\theta^{0}))^2\beta=1 \end{array}$}}
\sum \limits_{j = 1}^{n} \sum \limits_{i = 1}^{n}\left[ y_i - a_j -  b_j \beta^T(z_i - z_j)\right]^2 \omega_{ij} + \lambda \sum \limits_{j=1}^{n} \left|b_j\right| \sum \limits_{k=1}^{p}\beta_k,
\end{equation} $	
