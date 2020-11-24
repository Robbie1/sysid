from  control import  ss, step_response, dcgain
import matplotlib.pyplot as plt
import numpy as np

tss= 120
model = 'model.npz'
inputs = ['KOPC_SP', 'PIC100X_SP', 'TIC100_SP', 'WELL1.OP', 'WELL2.OP', 'WELL3.OP']
outputs = ['GAS.PV', 'HPCPOWER.PV', 'KOPC_OP', 'PIC100_OP', 'TIC100_OP']
mdl = np.load(model)
sys = ss(mdl['A'], mdl['B'], mdl['C'], mdl['D'],1)
gain_matrix = dcgain(sys).T
num_i = len(inputs)
num_o = len(outputs)
fig, axs = plt.subplots(num_i,num_o, figsize=(3*len(outputs), 2*len(inputs)), facecolor='w', edgecolor='k')
T = np.arange(tss)
for idx_i in range(num_i):
    for idx_o in range(num_o):
        ax = axs[idx_i][idx_o]
        t,y_step = step_response(sys,T, input=idx_i, output=idx_o)
        gain = round(gain_matrix[idx_i][idx_o],4)
        ax.plot(t, y_step,color='r')
        if idx_i == 0:
            ax.set_title(outputs[idx_o], rotation='horizontal', ha='center', fontsize=10)
        if idx_o == 0:
            ax.set_ylabel(inputs[idx_i], rotation=90, fontsize=10)
        ax.grid(color='k', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', colors='red',size=0,labelsize=4)
        ax.tick_params(axis='y', colors='red',size=0,labelsize=4)
        ax.annotate(str(gain),xy=(.72,.8),xycoords='axes fraction')
# fig.tight_layout()
plt.show()
