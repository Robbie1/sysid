from matplotlib import pyplot as plt
import matplotlib.dates as md
from sippy import functionsetSIM as fsetSIM
import numpy as np
from  control import  ss, step_response, dcgain

def plot_comparison(step_test_data, model, inputs, outputs, start_time, end_time, plt_input=False, scale_plt=False):
    """
    Plot the predicted and true output-signals.
    
    :param step_test_data: dataframe bject of loaded data.
    :param model: npz model file.
    :param inputs: Input vectors of the model.
    :param outputs: Output vectors of the model.
    :param start_time: Starting time of prediction data.
    :param end_time: Ending time of prediction data.
    :param plt_output: Boolean whether to Input vectors.
    :param scale_plt: Boolean whether to scale ouput vector plots.
    """
    
    val_data = step_test_data.loc[start_time:end_time]
    val_data.columns = [col[0] for col in val_data.columns]
    
    Time = val_data.index
    u = val_data[inputs].to_numpy().T
    y = val_data[outputs].to_numpy().T


    # Use the model to predict the output-signals.
    mdl = np.load(model)
    
    # The output of the model
    xid, yid = fsetSIM.SS_lsim_innovation_form(A=mdl['A'], B=mdl['B'], C=mdl['C'], D=mdl['D'], K=mdl['K'], y=y, u=u, x0=mdl['X0'])
    
    # Make the plotting-canvas bigger.
    plt.rcParams['figure.figsize'] = [25, 5]
    # For each output-signal.
    for idx in range(0,len(outputs)):
        plt.figure(idx)
        plt.xticks(rotation=15)
        plt.plot(Time, y[idx],color='r')
        plt.plot(Time, yid[idx],color='b')
        plt.ylabel(outputs[idx])
        plt.grid()
        plt.xlabel("Time")
        plt.title('output_'+ str(idx+1))
        plt.legend(['measurment', 'prediction'])
        ax=plt.gca()
        xfmt = md.DateFormatter('%m-%d-%yy %H:%M')
        ax.xaxis.set_major_formatter(xfmt)        
        if scale_plt==True:
            plt.ylim(np.amin(y[idx])*.99, np.amax(y[idx])*1.01)
        
    if plt_input == True:
        for idx in range(len(outputs), len(outputs) + len(inputs)):
            plt.figure(idx)
            plt.xticks(rotation=15)
            plt.plot(Time, u[idx-len(outputs)], color='r')
            plt.ylabel(inputs[idx-len(outputs)])
            plt.grid()
            plt.xlabel("Time")
            plt.title('input_'+ str(idx-len(outputs)+1))
            ax=plt.gca()
            xfmt = md.DateFormatter('%m-%d-%yy %H:%M')
            ax.xaxis.set_major_formatter(xfmt) 
    plt.show()

def plot_model(model, inputs, outputs, tss=90):
    """
    Plot the model matrix.

    :param model: npz model file.
    :param inputs: Input vectors of the model.
    :param outputs: Output vectors of the model
    :Param tss: time to steady state (length of x axis of subplot).
    """
    mdl = np.load(model)
    sys = ss(mdl['A'], mdl['B'], mdl['C'], mdl['D'],1)
    gain_matrix = dcgain(sys).T
    num_i = len(inputs)
    num_o = len(outputs)
    fig, axs = plt.subplots(num_i,num_o, figsize=(3*len(outputs), 2*len(inputs)), facecolor='w', edgecolor='k')
    fig.suptitle('Step responce: '+model)
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