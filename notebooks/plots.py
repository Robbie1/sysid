from matplotlib import pyplot as plt
from sippy import functionsetSIM as fsetSIM
import numpy as np

def plot_rediction(Time, u, y, yid, outputs, inputs, method, inc_output=False):
    for idx in range(0,len(outputs)):
        plt.figure(idx)
        plt.plot(Time, y[idx])
        plt.plot(Time, yid[idx])
        plt.ylabel(outputs[idx])
        plt.grid()
        plt.xlabel("Time")
        plt.title('output_'+ str(idx+1))
        plt.legend(['measurment', 'prediction, ' + method])
    if inc_output == True:
        for idx in range(len(outputs), len(outputs) + len(inputs)):
            plt.figure(idx)
            plt.plot(Time, u[idx-len(outputs)])
            plt.ylabel(inputs[idx-len(outputs)])
            plt.grid()
            plt.xlabel("Time")
            plt.title('input_'+ str(idx-len(outputs)+1))
    plt.show()

def plot_comparison(step_test_data, model, inputs, outputs, start_time, end_time, plt_input=False):
    """
    Plot the predicted and true output-signals.
    
    :param step_test_data: dataframe bject of loaded data.
    :param model: npz model file.
    :param inputs: Input vectors of the model.
    :param outputs: Output vectors of the model.
    :param start_time: Starting time of prediction data.
    :param end_time: Ending time of prediction data.
    :param plt_output: Boolean whether to Input vectors.
    """
    
    val_data = step_test_data.loc[start_time:end_time]
    val_data.columns = [col[0] for col in val_data.columns]
    
    Time = val_data.index
    u = val_data[inputs].to_numpy().T
    y = val_data[outputs].to_numpy().T


    # Use the model to predict the output-signals.
    mdl = np.load(model)
    
    # The output of the model
    xid, yid = fsetSIM.SS_lsim_process_form(mdl['A'], mdl['B'], mdl['C'], mdl['D'], u, mdl['X0'])
    
    # Make the plotting-canvas bigger.
    plt.rcParams['figure.figsize'] = [25, 5]
    # For each output-signal.
    for idx in range(0,len(outputs)):
        plt.figure(idx)
        plt.plot(Time, y[idx])
        plt.plot(Time, yid[idx])
        plt.ylabel(outputs[idx])
        plt.grid()
        plt.ylim(np.amin(y[idx])*.99, np.amax(y[idx])*1.01)
        plt.xlabel("Time")
        plt.title('output_'+ str(idx+1))
        plt.legend(['measurment', 'prediction'])
        
    if plt_input == True:
        for idx in range(len(outputs), len(outputs) + len(inputs)):
            plt.figure(idx)
            plt.plot(Time, u[idx-len(outputs)])
            plt.ylabel(inputs[idx-len(outputs)])
            plt.grid()
            plt.xlabel("Time")
            plt.title('input_'+ str(idx-len(outputs)+1))
    plt.show()