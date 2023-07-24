# Outshines Anonymous Documentation

This repository contains documents relevant to generating Outshines outlier detection scores. It will eventually live inside *the organization's* public Github repository along with the open source data. 

We also include a copy of the HTML files that were used in the expert evaluation surveys. 

Note that TODS has not been updated at the time of this commit to work on Python3.10. Please install [TODS](https://github.com/datamllab/tods) locally. 

## Running Outshines outlier score generation

First, create a file of test statistics by either using functions from the provided files (EWMA Functions, FlasH Modified functions, or TODS Functions, which also includes TODS scoring) or using your own. 
Then, if not using TODS, use the local or outshines scoring functions to obtain final outlier scores. 

```
python -m venv env
source env/bin/activate
#Optionally Download TODS
git clone https://github.com/datamllab/tods.git
cd tods
/usr/local/bin/python3.8 -m pip install -e . 
#Package Requirements
env/bin/python -m pip install -r requirements.txt
env/bin/python -m run.py 
```

Once you are finished with the code, you can deactivate the virtual environment
and (optionally) remove the environment itself.

```
deactivate
rm -r env
```

## Testing the code

We have not included testing code in this anonymous repository, but like all open source packages from the organization, this will be available open source. The output will show the number of unit tests that passed and failed, along with the percentage of code covered by the tests. None of the tests should fail and the code lines that are not covered by unit tests should be small and should not include critical sub-routines.