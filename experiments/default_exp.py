from mava.utils.training_utils import set_growing_gpu_memory
set_growing_gpu_memory()

import tensorflow as tf
from datetime import datetime
from mava.specs import MAEnvironmentSpec
from mava.utils.loggers import logger_utils
from acme.utils.loggers.base import Logger
from absl import app, flags
from selective_reincarnation_marl.iddpg import IDDPG
from selective_reincarnation_marl.mamujoco import Mujoco, get_mamujoco_args
from selective_reincarnation_marl.environment_loop import EnvironmentLoop


FLAGS = flags.FLAGS
flags.DEFINE_string("ragents", "0,1,2,3,4,5", "Comma seperated list of agent IDs to reincarnate or `None`.")
flags.DEFINE_string("seed", "0", "Random Seed.")
flags.DEFINE_string("dataset", "Medium", "`Good` or `Medium` or `Good-Medium`")

### MAIN ###
def main(_):
    env = Mujoco(
        get_mamujoco_args("6halfcheetah")
    )
    env_spec = MAEnvironmentSpec(env)
    agents = env_spec.get_agent_ids()

    timestamp = str(datetime.now())
    logger = logger_utils.make_logger(
        label="env_loop",
        directory="logs",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=timestamp,
        time_delta=1,  # log every 1 sec
    )

    train_logger = logger_utils.make_logger(
        label="trainer",
        directory="logs",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=timestamp,
        time_delta=1,  # log every 1 sec
    )

    if FLAGS.ragents == "None":
        ragents = []
    else:
        ragents = FLAGS.ragents.split(",")

    exploration_timesteps = {}
    teacher_dataset_training_steps = {}
    for i, agent in enumerate(agents):
        exploration_timesteps[agent] = 10_000.0 # always add exploration timesteps
        if str(i) in ragents:
            teacher_dataset_training_steps[agent] = 200_000
        else:
            teacher_dataset_training_steps[agent] = 0

    # Setup teacher dataset
    teacher_logdir = f"teacher_dataset/6halfcheetah/" # Good-Medium
    if FLAGS.dataset == "Good":
        teacher_logdir += "Good"
    elif FLAGS.dataset == "Medium": # NOTE!! Only Medium available in .zip
        teacher_logdir += "Medium"

    system = IDDPG(
        env_spec,
        logger,
        teacher_logdir=teacher_logdir,
        exploration_timesteps=exploration_timesteps,
        teacher_dataset_training_steps=teacher_dataset_training_steps,
        batch_size=64
    )

    env_loop = EnvironmentLoop(
        env,
        system,
        logger,
        train_logger,
        log_offline_data=False,
        max_timesteps=250_000 # 200_000 timesteps with teacher + 50_000 without
    )

    env_loop.run()

if __name__ == "__main__":
    app.run(main)