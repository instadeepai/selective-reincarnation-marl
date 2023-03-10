import copy
import tree
import reverb
import tensorflow as tf
import trfl
import sonnet as snt
from mava.adders import reverb as reverb_adders
import mava.components.tf.networks as mava_networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme import datasets
from selective_reincarnation_marl.offline_tools import MAOfflineSequenceDataset

def decay_schedule(timestep, max_timesteps):
    """Overwrite this to change teacher decay schedule."""
    if max_timesteps == 0.0 or timestep >= max_timesteps:
        return 0.0
    return 1.0

class IDDPG:

    def __init__(self, env_spec, logger, batch_size=32,
        max_replay_size=10_000, discount=0.99, lambda_=0.6, sequence_length=10, target_update_rate=0.005, sigma=0.1,
        checkpoint_path="checkpoints", teacher_logdir=None,
        exploration_timesteps={}, teacher_dataset_training_steps={},
        ):

        self.env_spec = env_spec
        self.agents = env_spec.get_agent_ids()
        self.logger = logger

        self.policy_networks = {}
        self.target_policy_networks = {}
        self.critic_networks = {}
        self.target_critic_networks = {}
        self.policy_optimizers = {}
        self.critic_optimizers = {}
        self.system_variables = {"policy_networks": {}, "critic_networks": {}}
        for agent in self.agents:
            dummy_obs = tf.zeros_like(env_spec.get_agent_specs()[agent].observations.observation)
            dummy_act = tf.zeros(shape=env_spec.get_agent_specs()[agent].actions.shape, dtype=env_spec.get_agent_specs()[agent].actions.dtype)
            dummy_state = tf.zeros_like(env_spec.get_extra_specs()["s_t"])
            dummy_critic_input = tf.concat([dummy_obs, dummy_state, dummy_act], axis=-1)

            dummy_obs = tf.expand_dims(dummy_obs, axis=0) # add batch dim
            dummy_critic_input = tf.expand_dims(dummy_critic_input, axis=0) # add batch dim

            self.action_dim = dummy_act.shape[-1]
            self.policy_networks[agent] = snt.DeepRNN(
                [
                    snt.Linear(64),
                    tf.nn.relu,
                    snt.GRU(64),
                    tf.nn.relu,
                    snt.Linear(self.action_dim),
                    tf.nn.tanh,
                ]
            )
            self.policy_networks[agent](
                dummy_obs,
                self.policy_networks[agent].initial_state(1)
            ) # initialise variables
            self.target_policy_networks[agent] = copy.deepcopy(self.policy_networks[agent])

            self.critic_networks[agent] = snt.Sequential(
                [
                    snt.Linear(64),
                    snt.LayerNorm(-1, False, False),
                    tf.nn.relu,
                    snt.Linear(64),
                    snt.LayerNorm(-1, False, False),
                    tf.nn.relu,
                    snt.Linear(1)
                ]
            )
            self.critic_networks[agent](tf.expand_dims(dummy_critic_input, axis=0)) # initialise variables
            self.target_critic_networks[agent] = copy.deepcopy(self.critic_networks[agent])

            self.policy_optimizers[agent] = snt.optimizers.Adam(3e-4)
            self.critic_optimizers[agent] = snt.optimizers.Adam(3e-4)

            self.system_variables["policy_networks"][agent] = self.policy_networks[agent].variables
            self.system_variables["critic_networks"][agent] = self.critic_networks[agent].variables

        # Exploration and teacher timesteps
        self.exploration_timesteps = {agent: exploration_timesteps[agent] if agent in exploration_timesteps else 0.0 for agent in self.agents}
        self.teacher_dataset_training_steps = {agent: teacher_dataset_training_steps[agent] if agent in teacher_dataset_training_steps else 0.0 for agent in self.agents}

        # Gaussian noise network for exploration
        specs = env_spec.get_agent_specs()
        agent_act_spec = list(specs.values())[
            0
        ].actions  # NOTE assume all agents have the same spec
        self.gaussian_noise_network = snt.Sequential(
            [
                mava_networks.ClippedGaussian(sigma),
                mava_networks.ClipToSpec(agent_act_spec),
            ]
        )
        
        # Hyper-params
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.sigma = sigma
        self.lambda_ = lambda_

        # Setup checkpointing
        self.checkpointer = tf2_savers.Checkpointer(
            directory=checkpoint_path,
            objects_to_save=self.system_variables,
            time_delta_minutes=1.0,
            add_uid=False,
            max_to_keep=10,
        )

        # Setup Reverb replay buffer
        adder_signiture = reverb_adders.ParallelSequenceAdder.signature(
            self.env_spec, sequence_length, {}
        )
        rate_limiter = reverb.rate_limiters.MinSize(1)
        replay_table = reverb.Table(
            name="priority_table",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=rate_limiter,
            signature=adder_signiture,
        )
        tables = [replay_table]
        self.replay_server = reverb.Server(tables=tables)
        self.replay_client = reverb.Client(f"localhost:{self.replay_server.port}")

        if teacher_logdir is not None:
            self.teacher_dataset = MAOfflineSequenceDataset(
                env_spec,
                logdir=teacher_logdir,
                batch_size=batch_size//2,
                shuffle_buffer_size=5000,
            )
        else:
            self.teacher_dataset = None

        # Setup dataset
        self.dataset = datasets.make_reverb_dataset(
            table="priority_table",
            server_address=self.replay_client.server_address,
            batch_size=batch_size,
        )
        self.dataset = iter(self.dataset)

        # # Setup adder to interface with Reverb
        self.adder = reverb_adders.ParallelSequenceAdder(
            client=self.replay_client,
            sequence_length=sequence_length,
            period=sequence_length-1,
        )

        # RNN core states
        self.core_states = {agent: self.policy_networks[agent].initial_state(1) for agent in self.agents}

        # Counters
        self.trainer_step_counter = 0.0
        self.timesteps_counter = 0.0

    def save_checkpoint(self):
        self.checkpointer.save(force=True)

    def observe_first(self, timestep, extras={}):

        # Re-initialise core states
        self.core_states = {agent: self.policy_networks[agent].initial_state(1) for agent in self.agents}

        self.adder.add_first(timestep, extras)

    def observe(self, actions, next_timestep, next_extras):
        self.adder.add(actions, next_timestep, next_extras)

    def select_actions(self, observations, evaluation=False):

        # Get agent actions
        actions, self.core_states = self._select_actions(
            observations, tf.convert_to_tensor(self.timesteps_counter), self.core_states, tf.convert_to_tensor(evaluation)
        )

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        # Increment counter
        self.timesteps_counter += 1

        return actions

    @tf.function
    def _select_actions(self, observations, timesteps, core_states, evaluation):
        actions = {}
        new_core_states = {}
        for agent in observations.keys():
            action, new_core_states[agent] = self._select_action(
                agent,
                observations[agent].observation,
                timesteps,
                core_states[agent],
                evaluation,
            )

            # Add exploration noise
            action = self.gaussian_noise_network(action)

            # Add to actions dict
            actions[agent] = action

        return actions, new_core_states

    def _select_action(self, agent, observation, timesteps, core_state, evaluation):
        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)

        # Pass observation through policy network
        action, core_state = self.policy_networks[agent](observation, core_state)
        
        if not evaluation:
            exploration_fraction = decay_schedule(timesteps, self.exploration_timesteps[agent])

            if tf.random.uniform((1,)) < exploration_fraction:
                action = tf.random.uniform((1, self.action_dim), -1.0, 1.0)
                # tf.print("RANDOM")

        return action, core_state

    def get_logs(self):

        logs = {}
        for agent in self.agents:
            exploration_fraction = decay_schedule(self.timesteps_counter, self.exploration_timesteps[agent])
            logs[f"{agent} Exploration Fraction"] = exploration_fraction

        return logs

    ### Training Methods ###

    def q_lambda_targets(self, rewards, env_discounts, target_max_qs):
        # Get time and batch dim
        B, T = rewards.shape[:2]

        # Make time major for trfl
        rewards = tf.transpose(rewards, perm=[1, 0])
        env_discounts = tf.transpose(env_discounts, perm=[1, 0])
        target_max_qs = tf.transpose(target_max_qs, perm=[1, 0])

        # Q(lambda)
        targets = trfl.multistep_forward_view(
            rewards[:-1],
            self.discount * env_discounts[:-1],
            target_max_qs[1:],
            lambda_=self.lambda_,
            back_prop=False,
        )

        # Make batch major again
        targets = tf.transpose(targets, perm=[1, 0])

        return targets

    @tf.function
    def _train(self, agents_to_train, student_sample, teacher_sample, trainer_step):
        logs = {}

        batch = sample_sequence_batch_agents(self.agents, student_sample, independent=True)

        if teacher_sample is not None:
            teacher_batch = sample_sequence_batch_agents(self.agents, teacher_sample, independent=True)
        else:
            teacher_batch = None

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = batch["discounts"]

        B, T = actions.shape[:2]

        for i, agent in enumerate(agents_to_train):
            teacher_dataset_fraction = decay_schedule(trainer_step, self.teacher_dataset_training_steps[agent])
            logs[f"{agent} Teacher Dataset Fraction"] = teacher_dataset_fraction

            if teacher_dataset_fraction > 0:
                agent_actions = tf.concat((teacher_batch["actions"][:,:,i], actions[:B//2,:,i]), axis=0)
                agent_observation = tf.concat((teacher_batch["observations"][:,:,i], observations[:B//2,:,i]), axis=0)
                agent_states = tf.concat((teacher_batch["states"], states[:B//2,:]), axis=0)
                agent_env_discounts = tf.concat((teacher_batch["discounts"][:,:,i], env_discounts[:B//2,:,i]), axis=0)
                agent_rewards = tf.concat((teacher_batch["rewards"][:,:,i], rewards[:B//2,:,i]), axis=0)
                # tf.print(f"Agent {i} TEACHER")
            else:
                agent_actions = actions[:,:,i]
                agent_observation = observations[:,:,i]
                agent_states = states
                agent_env_discounts = env_discounts[:,:,i]
                agent_rewards = rewards[:,:,i]
                # tf.print(f"Agent {i} STUDENT")
            
            agent_target_actions, _ = snt.static_unroll(
                self.target_policy_networks[agent],
                tf.transpose(agent_observation, perm=(1,0,2)),
                self.target_policy_networks[agent].initial_state(B),
            )
            agent_target_actions = tf.transpose(agent_target_actions, perm=(1,0,2))

            # Target critic
            target_critic_input = tf.concat((agent_observation, agent_states, agent_target_actions), axis=-1)
            agent_target_qs = self.target_critic_networks[agent](target_critic_input)

            # Compute Bellman targets
            # targets = agent_rewards[:,:-1] + self.discount * agent_env_discounts[:,:-1] * tf.squeeze(agent_target_qs)[:,1:]

            # Compute Q-lambda targets
            targets = self.q_lambda_targets(agent_rewards, agent_env_discounts, tf.squeeze(agent_target_qs))

            with tf.GradientTape(persistent=True) as tape:
                ### Critic Loss
                online_critic_input = tf.concat((agent_observation, agent_states, agent_actions), axis=-1)
                qs = self.critic_networks[agent](online_critic_input)

                # Mean Squared TD-Error
                critic_loss = 0.5 * (targets - tf.squeeze(qs)[:,:-1]) ** 2
                critic_loss = tf.reduce_mean(critic_loss)

                ### Policy Loss                
                online_actions, _ = snt.static_unroll(
                    self.policy_networks[agent],
                    tf.transpose(agent_observation, perm=(1,0,2)),
                    self.policy_networks[agent].initial_state(B),
                )
                online_actions = tf.transpose(online_actions, perm=(1,0,2))

                # Evaluate online action
                policy_critic_input = tf.concat([agent_observation, agent_states, online_actions], axis=-1)
                policy_qs = self.critic_networks[agent](policy_critic_input)
                policy_loss = -tf.reduce_mean(policy_qs)

            # Optimize critic
            variables = (
                *self.critic_networks[agent].trainable_variables,
            )
            gradients = tape.gradient(critic_loss, variables)
            gradients = tf.clip_by_global_norm(gradients, 10.0)[0]
            self.critic_optimizers[agent].apply(gradients, variables)

            # Optimize policy
            variables = (*self.policy_networks[agent].trainable_variables,)
            gradients = tape.gradient(policy_loss, variables)
            gradients = tf.clip_by_global_norm(gradients, 10.0)[0]
            self.policy_optimizers[agent].apply(gradients, variables)

            del tape # clear gradient tape

            # Update target networks
            online_variables = (
                *self.critic_networks[agent].variables,
                *self.policy_networks[agent].variables,
            )
            target_variables = (
                *self.target_critic_networks[agent].variables,
                *self.target_policy_networks[agent].variables,
            )   
            self.update_target_networks(
                online_variables,
                target_variables,
            )

            logs.update({
                f"{agent} Mean Q-value": tf.reduce_mean(qs),
                f"{agent} Critic Loss": critic_loss,
                f"{agent} Policy Loss": policy_loss,
            })

        return logs

    def update_target_networks(
        self, online_variables, target_variables
    ):
        """Update the target networks."""

        tau = self.target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

    def step(self):
        # Increment trainer step counter
        self.trainer_step_counter += 1

        # Sample student dataset
        sample = next(self.dataset)
        # additional_sample = next(self.dataset)

        # Teacher sample
        if self.teacher_dataset is not None:
            teacher_sample = next(self.teacher_dataset)
        else:
            teacher_sample = None

        # Pass sample to _train method
        logs = self._train(
            self.agents, sample, teacher_sample, trainer_step=tf.convert_to_tensor(self.trainer_step_counter)
        )

        # Add trainer steps to logs
        logs.update({"trainer_steps": self.trainer_step_counter})

        return logs

### UTILS ###

def sample_sequence_batch_agents(agents, sample, independent=False):
    # Unpack sample
    data = sample.data
    observations, actions, rewards, discounts, _, extras = (
        data.observations,
        data.actions,
        data.rewards,
        data.discounts,
        data.start_of_episode,
        data.extras,
    )

    all_observations = []
    all_legals = []
    all_actions = []
    all_rewards = []
    all_discounts = []
    all_logprobs = []
    for agent in agents:
        all_observations.append(observations[agent].observation)
        all_legals.append(observations[agent].legal_actions)
        all_actions.append(actions[agent])
        all_rewards.append(rewards[agent])
        all_discounts.append(discounts[agent])

        if "logprobs" in extras:
            all_logprobs.append(extras["logprobs"][agent])

    all_observations = tf.stack(all_observations, axis=2)  # (B,T,N,O)
    all_legals = tf.stack(all_legals, axis=2)  # (B,T,N,A)
    all_actions = tf.stack(all_actions, axis=2)  # (B,T,N,Act)
    all_rewards = tf.stack(all_rewards, axis=-1)  # (B,T,N)
    all_discounts = tf.stack(all_discounts, axis=-1)  # (B,T,N)

    if "logprobs" in extras:
        all_logprobs = tf.stack(all_logprobs, axis=2)  # (B,T,N,O)

    if not independent:
        all_rewards = tf.reduce_mean(all_rewards, axis=-1, keepdims=True)  # (B,T,1)
        all_discounts = tf.reduce_mean(all_discounts, axis=-1, keepdims=True)  # (B,T,1)

    # Cast legals to bool
    all_legals = tf.cast(all_legals, "bool")

    states = extras["s_t"] if "s_t" in extras else None  # (B,T,N,S)

    batch = {
        "observations": all_observations,
        "actions": all_actions,
        "rewards": all_rewards,
        "discounts": all_discounts,
        "legals": all_legals,
        "states": states,
    }

    if "logprobs" in extras:
        batch.update({"logprobs": all_logprobs})

    return batch

