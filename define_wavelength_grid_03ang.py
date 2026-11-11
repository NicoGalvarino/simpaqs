import numpy as np

blue = np.round(np.arange(start=3926.05, stop=4355.00, step=0.3), 2)

green = np.round(np.arange(start=5160.15, stop=5729.85+0.05, step=0.3), 2)

red = np.round(np.arange(start=6100.15, stop=6789.85+0.05, step=0.3), 2)

print('Blue:', blue.min(), blue.max())
print('Green:', green.min(), green.max())
print('Red:', red.min(), red.max())

all_arms = np.concatenate([blue, green, red])

np.save('/nfs/home/nguerrav/QSO_simpaqs/wavelen_grid_03ang.npy', all_arms)
np.save('/nfs/home/nguerrav/QSO_simpaqs/wavelen_grid_03ang_blue_arm.npy', blue)
np.save('/nfs/home/nguerrav/QSO_simpaqs/wavelen_grid_03ang_green_arm.npy', green)
np.save('/nfs/home/nguerrav/QSO_simpaqs/wavelen_grid_03ang_red_arm.npy', red)
