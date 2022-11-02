# ————————————————
# 版权声明：本文为CSDN博主「laizi_laizi」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/laizi_laizi/article/details/120935429
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('speed-eao2018.pdf')
plt.rc('font',family='Times New Roman')

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The Performance $vs.$ Speed on VOT-2018', fontsize=15)
ax.set_xlabel('Tracking Speed (FPS)', fontsize=15)
ax.set_ylabel('EAO', fontsize=15)


trackers = ['C-RPN', 'SiamVGG', 'DaSiamRPN', 'ATOM', 'SiamRPN++', 'DiMP', 'Ours (offline-2)', 'Ours (offline-1)', 'Ours (online)']
speed = np.array([50, 75, 65, 35, 50, 45, 72, 58, 25])
speed_norm = speed / 48
# speed_norm = np.array([50, 75, 65, 35, 50, 45, 72, 58, 25]) / 48
performance = np.array([0.273, 0.286, 0.380, 0.401, 0.414, 0.440, 0.438, 0.467, 0.490])

circle_color = ['cornflowerblue', 'deepskyblue',  'turquoise', 'gold', 'yellowgreen', 'orange', 'r', 'r', 'r']
# Marker size in units of points^2
volume = (300 * speed_norm/5 * performance/0.6)  ** 2

ax.scatter(speed, performance, c=circle_color, s=volume, alpha=0.4)
ax.scatter(speed, performance, c=circle_color, s=20, marker='o')
# text
ax.text(speed[0] - 2.37, performance[0] - 0.031, trackers[0], fontsize=10, color='k')
ax.text(speed[1] - 11.00, performance[1] - 0.005, trackers[1], fontsize=10, color='k')
ax.text(speed[2] - 3.5, performance[2] - 0.05, trackers[2], fontsize=10, color='k')
ax.text(speed[3] - 2.4, performance[3] - 0.032, trackers[3], fontsize=10, color='k')
ax.text(speed[4] - 2.9, performance[4] - 0.040, trackers[4], fontsize=10, color='k')
ax.text(speed[5] - 1.8, performance[5] - 0.042, trackers[5], fontsize=10, color='k')
ax.text(speed[6] - 6.0, performance[6] + 0.051, trackers[6], fontsize=14, color='k')
ax.text(speed[7] - 4.5, performance[7] -0.050, trackers[7], fontsize=12, color='k')
ax.text(speed[8] - 4, performance[8] -0.035, trackers[8], fontsize=12, color='k')

ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(20, 80)
ax.set_ylim(0.20, 0.53)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

# plot lines
ystart, yend = ax.get_ylim()
ax.plot([25, 25], [ystart, yend], linestyle="--", color='k', linewidth=0.7)
ax.plot([25, 58], [0.490,  0.467], linestyle="--", color='r', linewidth=0.7)
ax.plot([58, 72], [0.467, 0.438], linestyle="--", color='r', linewidth=0.7)
ax.text(26, 0.230, 'Real-time line', fontsize=11, color='k')

fig.savefig('speed-eao2018.svg')


pdf.savefig()
pdf.close()
plt.show()
