# symmetry-cs285

## Installation
1. Install pipenv:
   ```
   pip install pipenv --user
   ```
2. Create a pipenv environment:
   ```
   pipenv install
   ```
3. Install d3rlpy from the source
   ```
   cd d3rlpy
   pipenv run pip install -e .
   ```
4. Activate the environment
   ```
   pipenv shell
   ```
5. To run the notebook inside the environment:
   ```
   pipenv run jupyter notebook
   ```
6. To see the tensorboard logs:
   ```
   pipenv run tensorboard --logdir .
   ```
