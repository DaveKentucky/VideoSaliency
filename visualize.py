import numpy as np
import matplotlib.pyplot as plt


results = np.load('loss.npy')
n_epochs = results.shape[0]
x = np.linspace(1, n_epochs, n_epochs)
y = results[:, 2]

plt.plot(x, y)
# plt.title("Zmiana wartości funkcji straty podczas trenowania modelu")

plt.xlabel("Epoka")
plt.ylabel("NSS")

plt.savefig('plot_nss.png')
plt.show()
