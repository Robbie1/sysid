from sippy import *
import numpy as np
import pandas as pd
import plots

#load spteptest data from a TSV file
file = r'data\upstream_june_extracted.txt'
step_test_data = pd.read_table(file,header=[0, 1,2],index_col=[0], parse_dates=[0])

#slice data for model identification case
start = '2016-07-10 04:00:00'
end = '2016-07-21 00:00:00'
step_test = step_test_data.loc[start:end]
Time = step_test.index

#Drop mutilevel index
step_test.columns = [col[0] for col in step_test.columns]

#select Inputs and Outputs for the model identification case
inputs = ['KOPC_SP', 'PIC100X_SP', 'TIC100_SP', 'WELL1.OP', 'WELL2.OP', 'WELL3.OP']
outputs = ['GAS.PV', 'HPCPOWER.PV', 'KOPC_OP', 'PIC100_OP', 'TIC100_OP']

#Convert dataframe to numpy arry.
u = step_test[inputs].to_numpy().T
y = step_test[outputs].to_numpy().T
print('Output shape:', y.shape)
print('Input shape:',u.shape)

#specify model identification parameters, reffer the documentation for detais.
method='CVA'
IC = 'AICc' # None, AIC, AICc, BIC
TH = 30 # The length of time horizon used for regression
fix_ordr = 4 # Used if and only if IC = 'None'
max_order = 25 # Used if IC = AIC, AICc or BIC
req_D = False
force_A_stable = False

#Identify model
# Fit model
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
model = 'model.npz'
np.savez(model, A=sys_id.A, B=sys_id.B, C=sys_id.C, D=sys_id.D, K=sys_id.K, X0=sys_id.x0)

#Predict outputs uding identified model
start_time = '2016-06-21 04:00:00'
end_time = '2016-07-10 03:59:00'
plots.plot_comparison(step_test_data, model, inputs, outputs, start_time, end_time, plt_input=False, scale_plt=True)
# plots.plot_model(model, inputs, outputs)
