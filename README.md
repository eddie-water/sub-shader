# Create a virtual environment and activate 
I'm using WSL so I do:  
$ py -m venv .venv  
$ source .venv/bin/activate  

# Install the dependencies
$ py install -r requirements.txt

## If PyQt5 says something like:  
"AttributeError: module 'sipbuild.api' has no attribute 
'prepare_metadata_for_build_wheel'" 

$ pip install --upgrade pip setuptools wheel

# Run 
$ py main.py

# TODO Hierarchy
NOW > NEXT > SOON > LATER > EVENTUALLY