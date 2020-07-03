import visdom
import numpy as np

class LinePlotter(object):
    def __init__(self, env_name="main"):
        print("Connecting to the Visdom server. Make sure it is online by running 'python -m visdom.server'.")
        self.vis = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, x, y, y_label, var_name, title=""):
        window = title + " " + y_label
        if window not in self.plots:
            self.plots[window] = self.vis.line(
                                    X=np.array([x, x]),
                                    Y=np.array([y, y]),
                                    env=self.env,
                                    opts=dict(
                                        legend=[var_name],
                                        title=window,
                                        xlabel="Epochs",
                                        ylabel=y_label
                                        ))
        else:
            self.vis.line(X=np.array([x]),
                          Y=np.array([y]),
                          env=self.env,
                          win=self.plots[window],
                          name=var_name,
                          update = 'append')
