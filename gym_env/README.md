# Sim2Real-Compass-CodeBase

> DELETE the author info for anonymity purpose.


## Prepare Conda Env
```Shell
conda create -n compass python==3.8.16
conda activate compass


pip3 install torch torchvision torchaudio
pip3 install ipykernel
pip install matplotlib
# pip install gym
# pip install stable_baselines3
pip install seaborn
pip install imageio[ffmpeg]
pip install tqdm
pip install tensorboard h5py
```

## How to run
```Shell
# In Common
cd ./Sim2Real-Compass-CodeBase/scripts/
conda activate compass

## FOR Default Cusual DR + Dummy Agent
# Check `Sim2Real-Compass-CodeBase/scripts/utils/causual_DR_additional_Helper.py`
python3 main.py

## FOR Training SAC Policy
# Check `Sim2Real-Compass-CodeBase/scripts/utils/train_agent_additional_Helper.py`
python3 main.py --use_sac_agent
```


## How to see the logs!
> Ues google chrome !!! Note safari for the best experience.
```Shell
tensorboard --logdir=./Sim2Real-Compass-CodeBase/scripts/logdir/
```