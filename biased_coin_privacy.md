# $\epsilon$ for biased coin
Calculating the privacy guarantee for a biased coin:
Tell truth with probability $\theta$, and if random, flip a fair coin.

\[
\begin{align*}
P(y|T=y) &= P(\text{tell truth})\cdot1 + (1-P(\text{tell truth})) \cdot 1/2\\
&= \theta + (1-\theta)\cdot 0.5\\
P(n|T=y) &= P(\text{tell truth})\cdot0 + (1-P(\text{tell truth})) \cdot 1/2\\
&= (1-\theta)\cdot 0.5\\
\frac{P(y|T=y)}{P(n|T=y)}& = \frac{\theta + (1-\theta)\cdot 0.5}{(1-\theta)\cdot 0.5} \leq e^\epsilon\\
\epsilon &= \ln\left(\frac{\theta + (1-\theta)\cdot 0.5}{(1-\theta)\cdot 0.5}\right)
\end{align*}
\]