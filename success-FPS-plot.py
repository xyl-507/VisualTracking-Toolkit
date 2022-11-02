# ————————————————
# 版权声明：本文为CSDN博主「laizi_laizi」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/laizi_laizi/article/details/120935429
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages


# pdf = PdfPages('speed-eao2018.pdf')
plt.rc('font',family='Times New Roman')

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The Performance $vs.$ Speed on UAV123', fontsize=15)
ax.set_xlabel('Tracking Speed (FPS)', fontsize=15)
ax.set_ylabel('Success (%)', fontsize=15)


# trackers = ['ECO', 'ECO-HC', 'C-COT', 'DSiam', 'DaSiamRPN', 'SiamRPN', 'SiamFC', 'KCF', 'BACF', 'CSRDCF', 'SRDCF', 'ARCF', 'AutoTrack', 'Ours']
# speed = np.array([8, 60, 1, 45, 134, 160, 86, 95, 14.4, 58, 5.6, 15.3, 65.4, 100])
trackers = ['ECO', 'SiamTPN', 'C-COT', 'BASCF', 'SiamFC', 'LST', 'BACF', 'CSR-DCF', 'SRDCF', 'ARCF',
            'AutoTrack', 'SiamRPN++', 'CIFT', 'DAFSiamRPN', 'GlobalTrack', 'TGFAT(Ours)']
speed = np.array([8, 105, 1, 27.8, 86, 40.0, 14.4, 58, 5.6, 15.3, 65.4, 49, 87.0, 108.4, 11.3, 41.3])
speed_norm = speed / 48
# performance = np.array([53.7, 51.7, 50.2, 40.0, 56.9, 55.7, 49.4, 33.1, 46.1, 48.1, 46.4, 46.8, 47.2, 58.7])
performance = np.array([53.7, 59.3, 50.2, 50.5, 49.4, 45.9, 46.1, 48.1, 46.4, 46.8, 47.2, 61.2, 58.0, 54.7, 51.9, 61.7])
color = ['cornflowerblue', 'deepskyblue', 'turquoise', 'gold', 'yellowgreen', 'orange', 'sandybrown', 'mediumpurple', 'lightsalmon', 'lightsteelblue',
         'hotpink', 'lime','darkkhaki','cornflowerblue','lime', 'violet']
# shape = ['v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', '+', 'o']
# size = 50
# Marker size in units of points^2
volume = (2 * speed_norm/5 * performance/0.6)  ** 2
# volume = (speed_norm* performance)
# print(volume)

# 画形状
ax.scatter(speed, performance, c=color, s=volume, alpha=0.4)
ax.scatter(speed, performance, c=color, s=20, marker='o')
# ax.scatter(speed[0], performance[0], c=color[0], s=size, marker=shape[0])
# ax.scatter(speed[1], performance[1], c=color[1], s=size, marker=shape[1])
# ax.scatter(speed[2], performance[2], c=color[2], s=size, marker=shape[2])
# ax.scatter(speed[3], performance[3], c=color[3], s=size, marker=shape[3])
# ax.scatter(speed[4], performance[4], c=color[4], s=size, marker=shape[4])
# ax.scatter(speed[5], performance[5], c=color[5], s=size, marker=shape[5])
# ax.scatter(speed[6], performance[6], c=color[6], s=size, marker=shape[6])
# ax.scatter(speed[7], performance[7], c=color[7], s=size, marker=shape[7])
# ax.scatter(speed[8], performance[8], c=color[8], s=size, marker=shape[8])
# ax.scatter(speed[9], performance[9], c=color[9], s=size, marker=shape[9])
# ax.scatter(speed[10], performance[10], c=color[10], s=size, marker=shape[10])
# ax.scatter(speed[11], performance[11], c=color[11], s=size, marker=shape[11])

# text
ax.text(speed[0] - 2.9, performance[0] + 0.6, trackers[0], fontsize=10, color='k')
ax.text(speed[1] + 2.00, performance[1] + 3.0, trackers[1], fontsize=10, color='k')
ax.text(speed[2], performance[2] + 0.5, trackers[2], fontsize=10, color='k')
ax.text(speed[3] - 4, performance[3] - 3.0, trackers[3], fontsize=10, color='k')
ax.text(speed[4] + 6.5, performance[4] - 4, trackers[4], fontsize=10, color='k')
ax.text(speed[5] - 3, performance[5] + 3.2, trackers[5], fontsize=10, color='k')
ax.text(speed[6] - 4.0, performance[6] - 2, trackers[6], fontsize=10, color='k')
ax.text(speed[7] - 15.5, performance[7] - 3.0, trackers[7], fontsize=10, color='k')
ax.text(speed[8] - 5, performance[8] + 0.9, trackers[8], fontsize=10, color='k')
ax.text(speed[9], performance[9] + 0.91, trackers[9], fontsize=10, color='k')
ax.text(speed[10] - 5, performance[10] -4.0, trackers[10], fontsize=10, color='k')
ax.text(speed[11] - 7, performance[11] -4.5, trackers[11], fontsize=10, color='k')
ax.text(speed[12] - 9.5, performance[12] -1.5, trackers[12], fontsize=10, color='k')
ax.text(speed[13] - 10.5, performance[13] - 2.3, trackers[13], fontsize=10, color='k')
ax.text(speed[14], performance[14] + 0.6, trackers[14], fontsize=10, color='k')
# ax.text(speed[12] + 8, performance[12] -0.035, trackers[12], fontsize=10, color='k')
ax.text(speed[15] - 8, performance[15] + 2, trackers[15], fontsize=12, color='k')


ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(0, 125)
ax.set_ylim(30, 70)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

# plot lines
ystart, yend = ax.get_ylim()
ax.plot([30, 30], [ystart, yend], linestyle="--", color='k', linewidth=0.7)
# ax.plot([25, 58], [0.490,  0.467], linestyle="--", color='r', linewidth=0.7)
# ax.plot([58, 72], [0.467, 0.438], linestyle="--", color='r', linewidth=0.7)
ax.text(31, 33, 'Real-time line', fontsize=11, color='k')
# ax.plot([60, 60], [ystart, yend], linestyle="--", color='k', linewidth=0.7)
# ax.text(61, 33, 'High-time line', fontsize=11, color='k')

# fig.savefig('speed-eao2018.svg')
fig.savefig('speed-success-UAV123.jpg', dpi=600, bbox_inches='tight')


# pdf.savefig()
# pdf.close()
plt.show()
