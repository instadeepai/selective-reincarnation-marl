# 🤖🧟 Selective Reincarnation in MARL

Official repository for [Reduce, Reuse, Recycle: Selective Reincarnation in Multi-Agent Reinforcement Learning]() paper, accepted at the *Reincarnating RL* workshop at ICLR 2023.

## Abstract
> 'Reincarnation' in reinforcement learning has been proposed as a formalisation of reusing prior computation from past experiments when training an agent in an environment. In this paper, we present a brief foray into the paradigm of reincarnation in the multi-agent (MA) context. We consider the case where only some agents are reincarnated, whereas the others are trained from scratch -- selective reincarnation. In the fully-cooperative MA setting with heterogeneous agents, we demonstrate that selective reincarnation can lead to higher returns than training fully from scratch, and faster convergence than training with full reincarnation. However, the choice of which agents to reincarnate in a heterogeneous system is vitally important to the outcome of the training -- in fact, a poor choice can lead to considerably worse results than the alternatives. We argue that a rich field of work exists here, and we hope that our effort catalyses further energy in bringing the topic of reincarnation to the multi-agent realm.

## ⚠️ Warning!
Because of our dependency on [Mava](https://github.com/instadeepai/Mava), which depends on DeepMind's Reverb package, this code will likely only work on Linux systems. These instructions were tested on Ubuntu 22 with Python 3.8, using a Conda virtual environment.

## 💻 Installation

### 🐍 Create and activate conda environment
```
conda create -n srmarl python=3.8
conda activate srmarl
```

### 📚 Install Requirements
```
pip install -r requirements.txt
```

### 🐆 Install MAMuJoCo
```
bash install_mamujoco.sh
```

You will need to set the following environment variables. We recommend adding them to your [Conda activate file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables). Alternatively add them to your `.bashrc` file, or set them each time you open a new terminal.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:~/.mujoco/mujoco210/bin:/usr/lib/nvidia:/root/.mujoco/mujoco210/bin
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## 🧪 Run an experiment
Run an experiment by running the script `run.py`:

`python experiments/run_exp.py`

You can use the command line argument `--ragents` to specify which agents should be reincarnated, using a string of comma-separated agent IDs. For example:

`python experiments/run_exp.py --ragents=1,2,3`

`python experiments/run_exp.py --ragents=1,`

`python experiments/run_exp.py --ragents=2,3,5`

etc.

Use the string `None` to specify the whole system should be trained from scratch:

`python experiments/run_exp.py --ragents=None`


## 🤨 Troubleshooting

### ⛔️ Error:
After installing MAMuJoCo you get the following error:

`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.8.4 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
id-mava 0.1.3 requires numpy~=1.21.4, but you have numpy 1.24.2 which is incompatible.`
### ✅ Solution:
Just ignore this error. The code should still work properly.

### ⛔️ Error:
When you run the script `run.py`, you get the error message:
`ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`
### ✅ Solution: 
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib`

### ⛔️ Note:
Due to the small size of our neural networks, this code will run slower on a GPU than a CPU. You can easily run without GPU by setting the following environment variable:

`export CUDA_VISIBLE_DEVICES=""`