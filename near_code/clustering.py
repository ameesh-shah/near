from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from dtaidistance import clustering


""" s1 = a[0]
s2 = a[1]
d = dtw_ndim.distance(s1, s2)
 """

model = clustering.Hierarchical(dtw_ndim.distance_matrix, {'ndim':2})
model = clustering.KMedoids(dtw_ndim.distance_matrix, {'ndim':2}, k=3)


a = []
for datafile in ['data/ant_maze_left_train_paths', 'data/ant_maze_right_train_paths', 'data/ant_maze_top_train_paths']:
    temp = np.load(datafile, allow_pickle=True)[np.random.randint(0, 100, (5,))]
    a += temp.tolist()

#print(dtw_ndim.distance_matrix(a, ndim=2))

print(len(a), len(a[0]), len(a[1]), len(a[0][0]))
cluster_idx = model.fit(a)
print(cluster_idx)

import matplotlib.pyplot as plt 


for i,path in enumerate(a):
    path = np.array(path)
    plt.plot(path[:,0],path[:,1])
    plt.text(path[-1,0], path[-1,1], f'{i}')

    print(f"plotting path {path[0]} and {path[-1]}")
# savefig without bounding box
plt.savefig('data/ant_trajectories_noise.png', bbox_inches='tight', pad_inches=0)