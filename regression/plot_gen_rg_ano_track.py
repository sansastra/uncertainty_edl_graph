import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(6, 5))


try:
    with open("../resources/rostock_gedsar_interpol_1min.csv", "rb" ) as f:
        overall_data = np.array(pd.read_csv(f))

except IOError:
    print("Error: File does not appear to exist for track ")
    exit(0)

im = Image.open('../resources/world_11-4_54_12-2_54-4.PNG')  # in degrees and minutes
ax.imshow(im, extent=(11.6667, 12.3333, 54.0, 54.6667), aspect='auto')

######## Generate anomaly track ##############

traj_start = 2215 #2176
traj_end =   2261#2376
ano_data_left = overall_data[traj_start:traj_end, :].copy()
np.savetxt("gedsar_data.csv", ano_data_left, delimiter=",")
ax.plot(overall_data[:,0], overall_data[:,1], 'w.', markersize=2, label= "Normal Trajectories")

ind_start =  10 # 45
step = 15
for i in range(step+1):
    ano_data_left[ind_start + i, 0] = np.sqrt((overall_data[traj_start + ind_start + i, 0] - 0.0005 * (i ** 2)) ** 2)
    ano_data_left[ind_start + 2 * step - i, 0] = np.sqrt((overall_data[traj_start + ind_start + 2 * step - i, 0] - 0.0005 * (i ** 2)) ** 2)
ano_data_left[ind_start + step, 0] = ano_data_left[ind_start + step - 1, 0]
np.savetxt("gedsar_ano_left.csv", ano_data_left, delimiter=",")
ax.plot(ano_data_left[:, 0], ano_data_left[:, 1], 'r.', markersize=4, label="Anomalous Trajectories")

############# create anomaly to the right ##################
ano_data_right = overall_data[traj_start:traj_end, :].copy()

for i in range(step+1):
    ano_data_right[ind_start + i, 0] = np.sqrt((overall_data[traj_start + ind_start + i, 0] + 0.0005 * (i ** 2)) ** 2)
    ano_data_right[ind_start + 2 * step - i, 0] = np.sqrt((overall_data[traj_start + ind_start + 2 * step - i, 0] + 0.0005 * (i ** 2)) ** 2)
ano_data_right[ind_start + step, 0] = ano_data_right[ind_start + step - 1, 0]
np.savetxt("gedsar_ano_right.csv", ano_data_right, delimiter=",")
ax.plot(ano_data_right[:, 0], ano_data_right[:, 1], 'r.', markersize=4)



plt.pause(0.01)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
ax.legend()
plt.savefig("./results/rg_ano_track.pdf")
plt.savefig("./results/rg_ano_track.png")
plt.show()
plt.pause(0.01)

