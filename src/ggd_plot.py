import numpy as np
import scipy as sp
from scipy.stats import gennorm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

alpha_beta_list = [
    (1,0.5), (1,1), (1,1.5), (1,2),
    (0.5,0.5), (0.5, 2),
    (5, 0.5), (5,2)
]
def plot_ggd(alpha_beta_list):
    plt.figure(figsize=(15,9))
    for (alpha, beta) in alpha_beta_list:
        x = np.linspace(
            -3.5,3.5, 100
        )
        plt.plot(
            x, gennorm.pdf(x, beta=beta, loc=0, scale=alpha),
            '-', lw=5, label=r'$\alpha={}, \beta={}$'.format(alpha, beta)
        )
    plt.legend(fontsize=32)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig('ggd_all.png')
    plt.show()

plot_ggd(alpha_beta_list)