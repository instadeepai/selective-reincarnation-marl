import os
from typing import Any, Dict, Optional, Tuple
import dm_env
import numpy as np
import tensorflow as tf
import tree
from pathlib import Path
import matplotlib.pyplot as plt
import reverb
from mava.specs import MAEnvironmentSpec
from mava import types
from mava.types import OLT
from mava.adders.reverb.base import Step

# This code is adapted from OG-MARL https://sites.google.com/view/og-marl

def get_schema(env_spec):
    agent_specs = env_spec.get_agent_specs()

    schema = {}
    for agent in env_spec.get_agent_ids():
        spec = agent_specs[agent]

        schema[agent + "_observations"] = spec.observations.observation
        schema[agent + "_legal_actions"] = spec.observations.legal_actions
        schema[agent + "_actions"] = spec.actions
        schema[agent + "_rewards"] = spec.rewards
        schema[agent + "_discounts"] = spec.discounts
        schema[agent + "_next_observations"] = spec.observations.observation

    ## Extras
    # Global env state
    extras_spec = env_spec.get_extra_specs()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]
        schema["next_env_state"] = extras_spec["s_t"]

    schema["episode_return"] = np.array(0, dtype="float32")

    return schema


class WriteSequence:
    def __init__(self, schema, sequence_length):
        """An object to store a sequence to be written to disk."""
        self.schema = schema
        self.sequence_length = sequence_length
        self.numpy = tree.map_structure(
            lambda x: np.zeros(dtype=x.dtype, shape=(sequence_length, *x.shape)),
            schema,
        )
        self.t = 0

    def insert(self, agents, timestep, actions, next_timestep, extras):
        assert self.t < self.sequence_length
        for agent in agents:
            self.numpy[agent + "_observations"][self.t] = timestep.observation[
                agent
            ].observation

            self.numpy[agent + "_actions"][self.t] = actions[agent]

            self.numpy[agent + "_rewards"][self.t] = next_timestep.reward[agent]

            self.numpy[agent + "_discounts"][self.t] = next_timestep.discount[agent]

        ## Extras
        # Global env state
        if "s_t" in extras:
            self.numpy["env_state"][self.t] = extras["s_t"]

        # increment t
        self.t += 1

    def zero_pad(self, agents, episode_return):
        # Maybe zero pad sequence
        while self.t < self.sequence_length:
            for agent in agents:
                for item in [
                    "_observations",
                    "_actions",
                    "_rewards",
                    "_discounts",
                ]:
                    self.numpy[agent + item][self.t] = np.zeros_like(
                        self.numpy[agent + item][0]
                    )

                # Global env state
                if "env_state" in self.numpy:
                    self.numpy["env_state"][self.t] = np.zeros_like(
                        self.numpy["env_state"][0]
                    )

            # Increment time
            self.t += 1

        self.numpy["episode_return"] = np.array(episode_return, dtype="float32")

def get_sequence_schema(env_spec):
    environment_spec = env_spec
    agent_specs = environment_spec.get_agent_specs()

    schema = {}
    for agent in environment_spec.get_agent_ids():
        spec = agent_specs[agent]

        schema[agent + "_observations"] = spec.observations.observation
        schema[agent + "_actions"] = spec.actions
        schema[agent + "_rewards"] = spec.rewards
        schema[agent + "_discounts"] = spec.discounts

    ## Extras
    # Global env state
    extras_spec = environment_spec.get_extra_specs()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]

    schema["episode_return"] = np.array(0, dtype="float32")

    return schema


class MAOfflineEnvironmentSequenceLogger:
    def __init__(
        self,
        environment,
        sequence_length: int,
        period: int,
        logdir: str = "./offline_env_logs",
        min_sequences_per_file: int = 100,
    ):
        """A Wrapper for a multi-agent environment that stores sequences of interactions."""
        self._environment = environment
        env_spec = MAEnvironmentSpec(environment)
        self._schema = get_sequence_schema(env_spec)

        self._active_buffer = []
        self._write_buffer = []

        self._min_sequences_per_file = min_sequences_per_file
        self._sequence_length = sequence_length
        self._period = period

        self._logdir = logdir
        os.makedirs(logdir, exist_ok=True)

        self._timestep: Optional[dm_env.TimeStep] = None
        self._extras: Optional[Dict] = None
        self._episode_return = None

        self._num_writes = 0
        self._timestep_ctr = 0

    def reset(self) -> Tuple[dm_env.TimeStep, Dict]:
        """Resets the env and log the first timestep.

        Returns:
            dm.env timestep, extras
        """
        timestep = self._environment.reset()

        if type(timestep) == tuple:
            self._timestep, self._extras = timestep
        else:
            self._timestep = timestep
            self._extras = {}

        self._episode_return = np.mean(list(self._timestep.reward.values()))
        self._active_buffer = []
        self._timestep_ctr = 0

        return self._timestep, self._extras

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, Dict]:
        """Steps the env and logs timestep.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm.env timestep, extras
        """

        next_timestep = self._environment.step(actions)

        if type(next_timestep) == tuple and len(next_timestep) == 2:
            next_timestep, next_extras = next_timestep
        else:
            next_extras = {}

        self._episode_return += np.mean(list(next_timestep.reward.values()))

        # Log timestep
        self._log_timestep(
            self._timestep, self._extras, next_timestep, actions, self._episode_return
        )
        self._timestep = next_timestep
        self._extras = next_extras

        return self._timestep, self._extras

    def _log_timestep(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict,
        next_timestep: dm_env.TimeStep,
        actions: Dict,
        episode_return: float,
    ) -> None:
        if self._timestep_ctr % self._period == 0:
            self._active_buffer.append(
                WriteSequence(
                    schema=self._schema, sequence_length=self._sequence_length
                )
            )

        for write_sequence in self._active_buffer:
            if write_sequence.t < self._sequence_length:
                write_sequence.insert(
                    self._agents, timestep, actions, next_timestep, extras
                )

        if next_timestep.last():
            for write_sequence in self._active_buffer:
                write_sequence.zero_pad(self._agents, episode_return)
                self._write_buffer.append(write_sequence)
        if len(self._write_buffer) >= self._min_sequences_per_file:
            self._write()

        # Increment timestep counter
        self._timestep_ctr += 1

    def _write(self) -> None:
        filename = os.path.join(
            self._logdir, f"sequence_log_{self._num_writes}.tfrecord"
        )
        with tf.io.TFRecordWriter(filename, "GZIP") as file_writer:
            for write_sequence in self._write_buffer:

                # Convert numpy to tf.train features
                dict_of_features = tree.map_structure(
                    self._numpy_to_feature, write_sequence.numpy
                )

                # Create Example for writing
                features_for_example = tf.train.Features(feature=dict_of_features)
                example = tf.train.Example(features=features_for_example)

                # Write to file
                file_writer.write(example.SerializeToString())

        # Increment write counter
        self._num_writes += 1

        # Flush buffer and reset ctr
        self._write_buffer = []

    def _numpy_to_feature(self, np_array: np.ndarray):
        tensor = tf.convert_to_tensor(np_array)
        serialized_tensor = tf.io.serialize_tensor(tensor)
        bytes_list = tf.train.BytesList(value=[serialized_tensor.numpy()])
        feature_of_bytes = tf.train.Feature(bytes_list=bytes_list)
        return feature_of_bytes

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)

class MAOfflineSequenceDataset:
    def __init__(
        self,
        env_spec,
        logdir,
        batch_size=32,
        shuffle_buffer_size=1000,
        return_pytorch_tensors=False,
    ):
        """Load tfrecord files into a Dataset."""
        self._schema = get_sequence_schema(env_spec)
        self._spec = env_spec
        self._agents = self._spec.get_agent_ids()
        self._return_pytorch_tensors = return_pytorch_tensors

        file_path = Path(logdir)
        filenames = [
            str(file_name) for file_name in file_path.glob("**/*.tfrecord")
        ]
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self._no_repeat_dataset = filename_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
                self._decode_fn
            ),
            cycle_length=None,
            num_parallel_calls=2,
            deterministic=False,
            block_length=None,
        )

        self._dataset = (
            self._no_repeat_dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
            )
            .batch(batch_size)
            .repeat()
        )
        self._batch_size = batch_size

        self._dataset = iter(self._dataset)

    def _decode_fn(self, record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema
            ),
        )

        for key, item in self._schema.items():
            example[key] = tf.io.parse_tensor(example[key], item.dtype)

        observations = {}
        actions = {}
        rewards = {}
        discounts = {}
        extras = {}
        for agent in self._agents:
            observations[agent] = example[agent + "_observations"]
            actions[agent] = example[agent + "_actions"]
            rewards[agent] = example[agent + "_rewards"]
            discounts[agent] = example[agent + "_discounts"]

        # Make OLTs
        for agent in self._agents:
            observations[agent] = OLT(
                observation=observations[agent],
                legal_actions=tf.zeros((1,)),
                terminal=tf.zeros(
                    1, dtype="float32"
                ),  # TODO only a place holder for now
            )

        ## Extras
        # Global env state
        if "env_state" in example:
            extras["s_t"] = example["env_state"]

        # Start of episode
        start_of_episode = tf.zeros(
            1, dtype="float32"
        )  # TODO only a place holder for now

        # If "episode return" in example
        extras["episode_return"] = example["episode_return"]

        # Pack into Step
        reverb_sample_data = Step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            start_of_episode=start_of_episode,
            extras=extras,
        )

        # Make reverb sample so that interface same as in online algos
        reverb_sample_info = reverb.SampleInfo(
            key=-1, probability=-1.0, table_size=-1, priority=-1.0
        )  # TODO only a place holder for now

        # Rever sample
        reverb_sample = reverb.ReplaySample(
            info=reverb_sample_info, data=reverb_sample_data
        )

        return reverb_sample

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self._dataset)

        while list(sample.data.rewards.values())[0].shape[0] < self._batch_size:
            sample = next(self._dataset)

        return sample

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._dataset, name)

    def profile(self, filename):
        plt.clf()
        plt.tight_layout()
        all_returns = []
        for item in self._no_repeat_dataset:
            if "episode_return" in item.data.extras:
                all_returns.append(item.data.extras["episode_return"].numpy())
            else:
                rewards = list(item.data.rewards.values())[
                    0
                ]  # Assume all agents have the same reward
                undiscounted_return = tf.reduce_sum(rewards)
                all_returns.append(undiscounted_return.numpy())
        plt.xlabel("Episode Returns")
        plt.ylabel("Count")
        num_bins = 50
        plt.hist(all_returns, num_bins)
        plt.savefig(filename)
        return all_returns