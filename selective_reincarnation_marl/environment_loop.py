import numpy as np
from selective_reincarnation_marl.offline_tools import MAOfflineEnvironmentSequenceLogger

class EnvironmentLoop:
    """A parallel MARL environment loop (adapted from Mava)"""

    def __init__(
        self, 
        environment,
        system,
        logger,
        train_logger,
        train_period=1,
        max_timesteps=1e7,
        log_offline_data=False
    ):
        # store system and environment.
        self.environment = environment
        self.system = system
        self.logger = logger
        self.train_logger = train_logger


        # Maybe do offline logging
        if log_offline_data:
            self.environment = MAOfflineEnvironmentSequenceLogger(self.environment, 10, 9, logger._path("offline_logs"))

        # Hyper-params
        self.max_timesteps = max_timesteps
        self.train_period = train_period

        # Counters
        self.episode_counter = 0
        self.timesteps = 0

    def run(self):
        """Run"""
        logs = {}
        # eval_logs = self.evaluation()
        # logs.update(eval_logs)

        while self.timesteps < self.max_timesteps:

            # Reset any counts and start the environment.
            episode_steps = 0

            timestep, extras = self.environment.reset()

            # Make the first observation.
            self.system.observe_first(timestep, extras=extras)

            # For evaluation, this keeps track of the total undiscounted reward
            # for each agent accumulated during the episode.
            episode_returns = {}
            rewards = timestep.reward
            for agent, reward in rewards.items():
                episode_returns[agent] = reward

            # Run an episode.
            while not timestep.last():
                actions = self.system.select_actions(timestep.observation)

                # Step the environment
                timestep, extras = self.environment.step(actions)

                # Have the agent observe the timestep and let the actor update itself.
                self.system.observe(actions, timestep, extras)

                # Book-keeping.
                self.timesteps += 1
                rewards = timestep.reward
                for agent, reward in rewards.items():
                    episode_returns[agent] += reward

                # Maybe train
                if self.timesteps % self.train_period == 0 and self.timesteps > 256:
                    self.train_logger.write(self.system.step()) # Train step

            # Checkpoint
            self.system.save_checkpoint()

            # Book-keeping.
            self.episode_counter += 1

            # Collect the results for logging
            logs.update({
                "episode_length": episode_steps,
                "episode_return": np.mean(list(episode_returns.values())),
                "episodes": self.episode_counter,
                "timesteps": self.timesteps
            })

            logs.update(self.system.get_logs())

            self.logger.write(logs)

    def evaluation(self):
        timestep, extras = self.environment.reset()

        # Make the first observation.
        self.system.observe_first(timestep, extras=extras)

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        episode_returns = {}
        rewards = timestep.reward
        for agent, reward in rewards.items():
            episode_returns[agent] = reward

        # Run an episode.
        while not timestep.last():
            actions = self.system.select_actions(timestep.observation, evaluation=True)

            # Step the environment
            timestep, extras = self.environment.step(actions)

            # Have the agent observe the timestep and let the actor update itself.
            self.system.observe(actions, timestep, extras)

            # Book-keeping.
            rewards = timestep.reward
            for agent, reward in rewards.items():
                episode_returns[agent] += reward

        # Collect the results for logging
        logs = {}
        logs["eval_episode_return"] = np.mean(list(episode_returns.values()))

        return logs