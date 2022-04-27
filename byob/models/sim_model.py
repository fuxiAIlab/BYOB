from gym.spaces import Box

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

from byob.models.policy_net import PolicyNetwork

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TFBundleModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=((20 + 3) * 1,),
                 **kw):
        super(TFBundleModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(
            Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
            model_config, name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

        print(obs_space, action_space, num_outputs, model_config, name, true_obs_shape)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["mask"]

        # Compute the predicted action embedding [BATCH, MAX_ACTIONS]
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


class TorchBundleModel(DQNTorchModel):
    """PyTorch version of above TFMMModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=((20 + 3) * 1,),
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        # self.action_embed_model = TorchFC(
        #     Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
        #     model_config, name + "_action_embed")

        self.action_embed_model = PolicyNetwork(
            Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
            model_config, name + "_action_embed")

        # print(obs_space, action_space, num_outputs, model_config, name, true_obs_shape)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["mask"]

        # Compute the predicted action embedding [BATCH, MAX_ACTIONS]
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"],
            "user": input_dict["obs"]["user"],
            "seq": input_dict["obs"]["seq"],
            "pool": input_dict["obs"]["pool"],
            # "pos": input_dict["obs"]["pos"],
            "bundle": input_dict["obs"]["bundle"]
        })

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
