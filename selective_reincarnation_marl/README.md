# Reincarnating MARL

## Warning!! 
Because of our dependency on Mava, which depends on DeepMind's Reverb package, this code will probably only work on linux.
These instructions where tested on Ubuntu 22, Python 3.8 and using a conda virtual envrionment.

## Install Requirements
`pip install -r requirements.txt`

## Install MAMuJoCo
`bash install_mamujoco.sh`

You will need to set the following environment variables. We reccomend adding them to your [conda activate file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables). Alternativly add them to your .bashrc file or set them each time you open a new terminal.

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:~/.mujoco/mujoco210/bin:/usr/lib/nvidia:/root/.mujoco/mujoco210/bin`

`export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so`

## Run an experiment
Note that due to the file size constraint of the supplimentary material we could only include the `Medium` dataset. But the system will still demonstrate good results on this dataset, but they wont be the same as the results in the paper.

Run an experiment by running the script `run.py`:

`python run.py`

You can use the command line argument `--ragents` to specify which agents should be reincarnated. 
Specify which agents should be reincarnated by a string og agent IDs seperated by commas. For example:

`python run.py --ragents=1,2,3`

or

`python run.py --ragents=1,`

or

`python run.py --ragents=2,3,5`

etc.

Use the string `None` to specify the whole system should be trained from scratch.

`python run.py --ragents=None`

## Troubleshooting

### Error:
After installing MAMuJoCo you get the following error:
`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.8.4 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
id-mava 0.1.3 requires numpy~=1.21.4, but you have numpy 1.24.2 which is incompatible.`
### Soultion:
Just ignore the error. The code should still work properly.

### Error:
When you run the script `run.py` you get the error message:
`ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`
### Solution: 
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib`

### Another warning!
There is a problem in our code that makes it run slow on GPU. We are working on resolving it. For now, its reccomended to train using CPU only, as its faster. You can easily run without GPU by setting the following environment variable.

`export CUDA_VISIBLE_DEVICES=""`