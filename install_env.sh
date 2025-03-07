# Step 1: Create conda environment
conda create -n sim2real-capture python==3.8.16 -y
conda activate sim2real-capture

# Step 2: Install packages

# Install pytorch (follow the instructions from https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other packages with pip
pip3 install ipykernel
pip install matplotlib
pip install seaborn
pip install imageio[ffmpeg]
pip install tqdm
pip install tensorboard h5py
pip3 install setuptools==50.3.1
pip3 install wheel==0.35.1
pip install gym==0.21.0
pip install stable_baselines3==1.8.0

# Setup robosuite
cd ./gym_env
pip install -e .
cd ..

# Install other packages
pip install transformers==4.5.1
pip install wandb
pip install packaging==20.4


echo "Installation and setup complete."
