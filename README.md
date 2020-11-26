# SysID: System identification

This is a collection of Jupyter notebooks and scripts to demonstrate the capablities of an opensource system identification package [SIPPY](https://github.com/CPCLAB-UNIPI/SIPPY). A comparison study of MATLAB system identification toolbox vs SIPPY can be found [here](https://github.com/jamestjsp/sysid/blob/main/docs/papers/An_open-source_System_Identification_Package_for_multivariable_processes.pdf) 

## Installation

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Open conda prompt and install Jupyter using the following command.

```cmd
conda install -c conda-forge jupyterlab
```
3. Create control conda [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the following command. You can use my yml file from [here](https://github.com/jamestjsp/sysid/blob/main/environment.yml).

```cmd
conda env create -f environment.yml
```
4. Verify that the new environment was installed correctly.
```cmd
conda env list
```
This should return something similar as given below:
```cmd
base                  *  C:\Miniconda3
control                  C:\Miniconda3\envs\control
```
5. Activate the new environment.
```cmd
conda activate control
```
6. To use Jupyter in conda env run the following command.
```cmd
python -m ipykernel install --user --name=control
```
7. Clone SIPPY from Github.
 * Colne the SIPPY from Github by running the following command (You should have git installed).
```bash
git clone "https://github.com/CPCLAB-UNIPI/SIPPY"
```
 * Else, download it from [Github](https://github.com/CPCLAB-UNIPI/SIPPY), then extract the zip file to a local folder.
8. To install SIPPY to control env, open conda prompt and *cd (change directory command)*  to the SIPPY folder then run the following commands.
```cmd
conda activate control
```
Then
```cmd
python setup.py install
```
9. To test, follow the below stpes.
   * *cd* to Example folder folder within SIPPY folder. Then run the following command.
Then
```cmd
python SS.py
```
This should show a trend of prediction after running a N4SID (As configured in the script).

10. Clone my [SysID](https://github.com/jamestjsp/sysid)
 * Colne the SIPPY from Github by running the following command (You should have git installed).
```bash
git clone "https://github.com/jamestjsp/sysid"
```
* Else, download it from [Github](https://github.com/jamestjsp/sysid), then extract the zip file to a local folder.

## Usage
1. Open conda prompt then activate control env. 
```cmd
conda activate control
```
2. *cd* to SysID folder then, launch Jupyter lab by running following command.

```cmd
jupyter lab
```
This should open Jupyter Lab in your browser then open notebook folder and try play the files *model_id_democol.ipynb* and *model_id_uptream_june_dataset_HP_section.ipynb*

## Documents and reference
SIPPY [document](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/user_guide.pdf) assumes that you have a good understanding about system identification. When I started with this library I struggled a little and [Aspentech whitepaper](https://github.com/jamestjsp/sysid/blob/main/docs/HOW%20TO%20USE%20SUBSPACE%20ID%20-%20AspenTech%20White%20Paper.pdf) helped me to pickup this useful library. 
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
