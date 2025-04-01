# CAPTURE
Dynamics as Prompts: In-Context Learning for Sim-to-Real System Identifications

IEEE Robotics and Automation Letters (RA-L), 2025

[[Webpage]](https://sim2real-capture.github.io/) | [[Arxiv]](https://arxiv.org/abs/2410.20357)

Please raise an issue or reach out to the authors if you need help with running the code.

## Prepare Conda Env
#### Note: Our simulation environments are modified from [robosuite](https://github.com/ARISE-Initiative/robosuite).
You can follow the bash file to setup the environment. 
```Shell
    ./install_env.sh
```

In order to collect dataset and train the transformer models, you can run: 
```Shell
    ./bash/data_collection.sh
    ./bash/model_training.sh
```

## Acknowledgement 
[Robosuite: A Modular Simulation Framework and Benchmark for Robot Learning](https://robosuite.ai/) <br>
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## Citation
```
@article{zhang2025dynamics,
  title={Dynamics as Prompts: In-Context Learning for Sim-to-Real System Identifications},
  author={Zhang, Xilun and Liu, Shiqi and Huang, Peide and Han, William Jongwon and Lyu, Yiqi and Xu, Mengdi and Zhao, Ding},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
