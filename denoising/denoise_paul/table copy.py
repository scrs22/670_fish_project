import numpy as np
import matplotlib.pyplot as plt


data = [[ 1623, 0.834,  0.66, 0.668],
        [ 481, 0.712, 0.873, 0.796],
        [ 298, 0.979, 0.983, 0.988],
        [ 6, 1, 0, 0.00896],
        [ 108, 0.698, 0.454, 0.534],
        [ 592, 0.767, 0.782, 0.786],
        [ 138, 0.851, 0.867, 0.898]
        ]

columns = ('Labels', 'P', 'R', 'mAP@.5')
rows = ['all', 'scallop', 'herring', 'dead-scallop', 'flounder', 'roundfish','skate']


the_table = plt.table(cellText=data,
                      rowLabels=rows,
                      colLabels=columns)

# plt.title('Approach 1: Uncropped Dataset')

plt.show()