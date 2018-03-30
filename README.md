# sysid_l2fly
System Identification, Modeling, and Control of Linear and Nonlinear Systems

## Usage
Currently working on controller and online modeling.

***NOTE: *** The files below needs to be revised and are currently inside Archive.

predict_gtm.py:  
Loads previously collected data from NASA GTM Matlab sim and uses deep learning to predict model dynamics. It uses 4 states (beta, p, q, phi) and 2 controls (?).

main.py:  
This file loads a previously trained model (saved inside 'models' folder) and a tries to control the plant using a modified version of iLQR controller.

predict_gtm_full.py:  
Loads previously collected data from NASA GTM Matlab sim and uses deep learning to predict model dynamics. It uses 12 states (beta, p, q, phi) and (?) controls (?).

## Installation

All the packages and dependencies are being managed using Anaconda3.
Please go ahead and [install Anaconda3](https://www.continuum.io/downloads#linux) (Python 3.5 or greater).

Clone this repository. Go to its main folder and run:
```
./setup_linux.sh
```

Everything should be installed by now. If there's any problem during the installation, please contact Vinicius Goecks at viniciusguigo@gmail.com.

Before using SysId_L2Fly, activate the environment:
```
source activate sysid_l2fly
```

You might want to deactivate it eventually:
```
source deactivate
```

## More Info / Questions

Jack Han-Hsun Lu: ineffable1201@gmail.com  
Vinicius G. Goecks: viniciusguigo@gmail.com
