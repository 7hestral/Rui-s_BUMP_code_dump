conda create --name myvenv
source activate myvenv
python -m pip install --upgrade pip --user

conda install -c anaconda ipykernel
pip install -r ./QPrism/requirements.txt --user
python -m ipykernel install --user --name=myvenv