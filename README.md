# Trigger with machine learning

Repository for the UniOvi code for this.


Contributors:
- Santiago Folgueras
- Pietro Vischia
- Artur Kalinowski
- Pelayo Leguina


To generate the proper environment: 
```
virtualenv pyenv  --python=3.11
source <name_of_venv>/bin/activate
pip install .
```

If you have python 3.11 already installed in your system, you can also run `python3.11 -m venv pyenv` instead of the `virtualenv` command above.

Dependencies and requirements: 
The list of dependencies can be seen in requirements.txt and installed by runing: 

``` 
pip install -r requirements.txt
```

In Lxplus, this line seems to have all the required dependencies
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
```