# conda create -n codeformer_deca python=3.7 -y
# conda activate codeformer_deca
pip install -r requirements_codeformer.txt 
pip install -r requirements_deca.txt
pip install pydantic==1.10.8 typing-extensions==4.5.0 gradio==3.34.0 openai pillow matplotlib_inline ipython
pip install pytorch3d==0.2.5
pip uninstall -y torch torchvision
conda install --yes pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install --yes pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install dlib
python basicsr/setup.py develop

