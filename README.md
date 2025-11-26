# Dataset
```legacy_code_generator```
src: https://github.com/ds4dm/learn2branch/tree/master

# Requirements.txt
```pip install -r requirements.txt```

To generate dataset, please also follow the guidance in ```legacy_code_generator```.

To play with ```disjunctive_dual``` and use LTR losses, please also ```pip install pytorchltr```. 
If an error occurred, it may be caused by python version > 3.10. To resolve this, a solution is

```git clone https://github.com/rjagerman/pytorchltr.git```

Go to ```setup.py``` and replace ```ext_modules= get_svmrank_parser_ext()``` with ```ext_modules= []```, then ```pip install .```