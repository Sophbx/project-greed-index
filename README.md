# project-greed-index
In the financial market, trend-following (momentum) strategies have long been proven to have excess return potential. However, market trends are often accompanied by short-term pullback risks during overheating phases, and momentum strategies are prone to pullbacks during this phase. At the same time, investor sentiment fluctuations, especially greed-driven pursuit of high prices, often further amplify the risk exposure at the end of the trend. Therefore, building an early warning model that can quantify momentum status and greed in real time and predict the probability of short-term retracements is of great value in improving strategy risk control and return quality.

Main Theme:
1. Market gains are often accompanied by increased momentum (new highs in prices, increased trading volume, and convergence of volatility)
2. But when the momentum is too strong and the emotions are too hot, it is easy to form a short-term mean reversion / pullback
3. We hope to quantitatively define this greedy state and identify the trend end signal in advance.
4. Application scenarios: risk control (active reduction of positions), reversal strategy (shorting or reducing long positions), dynamic adjustment of momentum strategy exposure
