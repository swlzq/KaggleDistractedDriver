# @Author:langyi
# @Time  :2019/4/1

import os
import pandas as pd
import visdom
import collections


# Show data distribution.
def data_visualize():
    driver_imgs_num = collections.OrderedDict()
    img_path = '/userhome/data/state-farm-distracted-driver-detection'
    data_frame = pd.read_csv(os.path.join(img_path, 'driver_imgs_list.csv'))
    drivers_labels = data_frame['subject'].drop_duplicates().values
    for label in drivers_labels:
        print(label)
        df = data_frame[data_frame['subject'] == label]
        count = len(df)
        driver_imgs_num[label] = count

    X = []
    Y = []
    for label in driver_imgs_num:
        print(label, ':', driver_imgs_num[label])
        X.append(label)
        Y.append(driver_imgs_num[label])

    vis = visdom.Visdom(env='data')
    vis.bar(
        X=Y,
        opts=dict(
            rownames=X,
            colormap='red'
        )
    )


if __name__ == '__main__':
    data_visualize()
