import numpy as np, matplotlib.pyplot as plt
from ltc import LTC

N = 50
Xs = [((i/N)-.5)*2 for i in range(N)]
Xs = [(np.sin(x*10)) for x in Xs] # X becomes a function of t

ltc = LTC()
ltc.fit(Xs)

plt.plot(Xs)
plt.plot(ltc.evaluate(Xs))

plt.show()
