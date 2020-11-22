from sippy import *
import numpy as np
import pandas as pd
import plots

#load spteptest data from a TSV file
file = r'data\upstream_june_extracted.txt'
step_test_data = pd.read_table(file,header=[0, 1,2],index_col=[0], parse_dates=[0])

#slice data for model identification case
start = '2016-07-01 04:00:00'
end = '2016-07-04 03:59:00'
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

#Identify model
method='CVA'
sys_id = system_identification(
    y, 
    u, 
    method,
    SS_fixed_order=25,
    SS_f=120,
#     SS_p=120,
    SS_D_required=False,
    SS_A_stability=False)
print('model order', sys_id.n)

#save model parameters A, B, C,D and X0 as npz file
model = 'model.npz'
np.savez(model, A=sys_id.A, B=sys_id.B, C=sys_id.C, D=sys_id.D, X0=sys_id.x0)

#Predict outputs uding identified model
start_time = '2016-06-24 04:00:00'
end_time = '2016-07-21 04:00:00'
plots.plot_comparison(step_test_data, model, inputs, outputs, start_time, end_time, plt_input=False)
