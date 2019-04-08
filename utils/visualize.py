# @Author:langyi
# @Time  :2019/3/27

# Encapsulate base method of visdom

import visdom
import time
import numpy as np


class Visualize(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # Number of point, i.e. x-axis point
        # e.g. ('loss',23), i,e 23th point
        self.index = {}
        self.log_text = ''
        assert self.vis.check_connection()

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        '''

        Plot (name,y) in name's current index
        '''
        x = self.index.get(name, 0)  # Get current index by name
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1  # index add 1

    def plot2(self, name, arr, legend, **kwargs):
        x = self.index.get(name, 0)
        assert len(arr) == 2, 'Y dim must be 2'
        Y = np.column_stack((np.array(arr[0]), np.array(arr[1])))
        X = np.column_stack((np.array([x]), np.array([x])))
        self.vis.line(
            Y=Y,
            X=X,
            win=name,
            opts=dict(title=name,
                      legend=legend),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        '''

        Log info
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info)
        )
        self.vis.text(self.log_text, win)

    def __getattr__(self, item):
        return getattr(self.vis, item)
