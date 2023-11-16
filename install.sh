conda create --name symmetry python=3.10
source activate
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install pip
pip install -r requirements.txt
cd d3rlpy
pip install -e .
cd ..

