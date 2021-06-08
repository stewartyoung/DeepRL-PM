import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import logging


class LivePlotNotebook(object):
    """
    Live plot using `%matplotlib notebook` in jupyter

    Usage:
    liveplot = LivePlotNotebook(labels=['a','b'])
    x = range(10)
    ya = np.random.random((10))
    yb = np.random.random((10))
    liveplot.update(x, [ya,yb])
    """

    def __init__(self, log_dir="plots/", episode=0, labels=[], title='', ylabel='returns', colors=None, linestyles=None, legend_outside=True,
        filename="", timestamp="", proportion=1.0, repeat=0, ESGeps=0.0):
        if not matplotlib.rcParams['backend'] == 'nbAgg':
            logging.warn("The liveplot callback only work when matplotlib is using the nbAgg backend. Execute 'matplotlib.use('nbAgg', force=True)'' or '%matplotlib notebook'")
        self.log_dir = log_dir
        if log_dir:
            try:
                os.mkdir(log_dir)
            except OSError:
                pass
        self.i = episode
        # store in object for more descriptive plot filename
        self.title = title
        self.filename = filename
        self.timestamp = timestamp
        self.proportion = proportion
        self.repeat = repeat
        self.ESGeps = ESGeps
        self.title = title
        fig, ax = plt.subplots(1, 1)

        for i in range(len(labels)):
            ax.plot(
                [0] * 20,
                label=labels[i],
                alpha=0.75,
                lw=2,
                color=colors[i] if colors else None,
                linestyle=linestyles[i] if linestyles else None,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('date')
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.set_title(title)

        # give the legend it's own space, the right 20% where it right align left
        if legend_outside:
            fig.subplots_adjust(right=0.8)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        else:
            ax.legend()
        self.ax = ax
        self.fig = fig

    def update(self, x, ys):
        x = np.array(x)

        for i in range(len(ys)):
            # update price
            line = self.ax.lines[i]
            line.set_xdata(x)
            line.set_ydata(ys[i])

        # update limits
        y = np.concatenate(ys)
        y_extra = y.std() * 0.1
        if x.min() != x.max():
            self.ax.set_xlim(x.min(), x.max())
        if (y.min() - y_extra) != (y.max() + y_extra):
            self.ax.set_ylim(y.min() - y_extra, y.max() + y_extra)
        self.fig.savefig(self.log_dir+
            self.timestamp+ "_"+
            self.filename+"_"+
            self.title+"_prop"+
            str(self.proportion)+
            "_esgeps"+str(self.ESGeps)+
            "_repeat"+str(self.repeat)+".png")
        self.fig.canvas.draw()
        self.i += 1
        