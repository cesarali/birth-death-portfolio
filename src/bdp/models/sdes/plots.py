import numpy as np
from matplotlib import pyplot as plt


def plot_only_arrivals(arrivals,support_border=None,save_plot=None):
    fig, ax = plt.subplots(1,1,figsize=(12,1.5))
    ax.vlines(arrivals, ymin=-0.1, ymax=1.1, color="#F88017",alpha=.9, label="Packed Arrivals")
    if support_border is None:
        plot_limit = (np.min(arrivals), np.max(arrivals))
    else:
        plot_limit = support_border
    ax.set_xlim(plot_limit)
    ax.set_ylim(0.,1.)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(r'Time ($t$)', fontsize=12)
    fig.tight_layout()
    if save_plot==None:
        plt.show()
    plt.close()