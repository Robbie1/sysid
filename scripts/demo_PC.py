
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sippy import *
import plots
# %
#load spteptest data from a TSV file
file = r'data\PC_Data_shifted.csv'
step_test_data = pd.read_csv(file,index_col='Time', parse_dates=True, skiprows=[1,2])
ts = pd.Timedelta(step_test_data.index[1] - step_test_data.index[0]).total_seconds() # data sampling time
#slice data for model identification case
start = '09/30/2012 09:00:00'
stop = '09/30/2012 12:20:10'
idinput = step_test_data.loc[start:stop].copy()

#select Inputs and Outputs for the model identification case
inputs = ['Fuel','Fan', 'Feed']
outputs = ['Temp','O2']

# prepare detrending filter 
tss = 5 * 60
mult_factor = 3
filt_tss = tss * mult_factor
numtaps = 32+1
cutoff = 2/filt_tss/mult_factor
fs = ts
pass_zero= 'lowpass'
window = ('exponential', None, 3)
coef = signal.firwin(numtaps=numtaps, cutoff=cutoff, window=window, pass_zero=pass_zero, fs=fs)
plots.plot_freuency_response(coef)
trend = signal.filtfilt(coef, 1.0,idinput, axis=0)
idinput = idinput - trend
# for column in idinput:
#     t = np.arange(len(idinput))
#     orginal_sig = idinput[column].copy()
#     trend = signal.filtfilt(coef, 1.0,idinput[column])
#     idinput[column] = idinput[column] - trend
#     plt.plot(t, orginal_sig, t, trend, t, idinput[column])
#     plt.legend(['orginal signal', 'trend', 'detrended signal'])
#     plt.show()

# idinput.plot(subplots=True)
# plt.show()
u = idinput[inputs].to_numpy().T
y = idinput[outputs].to_numpy().T
print('Output shape:', y.shape)
print('Input shape:',u.shape)

#specify model identification parameters, reffer the documentation for detais.
model = 'Precalciner.npz' #model file name
method='CVA'
IC = 'AIC' # None, AIC, AICc, BIC
TH = 30 # The length of time horizon used for regression
fix_ordr = 35 # Used if and only if IC = 'None'
max_order = 25 # Used if IC = AIC, AICc or BIC
req_D = False
force_A_stable = False

sys_id = system_identification(
    y, 
    u, 
    method,
    SS_fixed_order=fix_ordr,
    SS_max_order=max_order,
    IC=IC,
    SS_f=TH,
    SS_p=TH,
    SS_D_required=req_D,
    SS_A_stability=force_A_stable
    )

#print model order
# print('Model order:', sys_id.n)

#save model parameters A, B, C,D and X0 as npz file
np.savez(model, A=sys_id.A, B=sys_id.B, C=sys_id.C, D=sys_id.D, K=sys_id.K, X0=sys_id.x0)
plots.plot_model(model, inputs, outputs, tss=tss)

start_time = start
end_time = stop
plots.plot_comparison(step_test_data, model, inputs, outputs, start_time, end_time, plt_input=False)