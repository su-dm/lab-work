import gymnasium
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class SimpleLSTMRLModule(TorchRLModule, ValueFunctionAPI):
    """
    Simple single LSTM RLModule for comparison with the ensemble.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent class first
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initialize the model architecture."""
        # Get model configuration
        model_config = self.model_config.get("model_config_dict", {})

        # LSTM architecture parameters
        self.hidden_size = model_config.get("hidden_size", 128)
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config.get("dropout", 0.0)
        self.bidirectional = model_config.get("bidirectional", False)

        # Input size from observation space
        obs_space = self.observation_space
        if isinstance(obs_space, gymnasium.spaces.Box):
            input_size = int(np.prod(obs_space.shape))
        else:
            input_size = obs_space.n

        # Create single LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Get output size from the LSTM
        num_directions = 2 if self.bidirectional else 1
        lstm_output_size = self.hidden_size * num_directions

        # Action space size
        action_space = self.action_space
        if isinstance(action_space, gymnasium.spaces.Discrete):
            num_outputs = action_space.n
        else:
            num_outputs = int(np.prod(action_space.shape))

        # Policy head (action logits for discrete, mean for continuous)
        self.pi = nn.Linear(lstm_output_size, num_outputs)

        # Value head (state value estimation)
        self.vf = nn.Linear(lstm_output_size, 1)

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for inference (action selection)."""
        obs = batch["obs"]

        # Flatten observation if needed and add sequence dimension
        if len(obs.shape) == 2:
            # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)

        # Forward pass through the LSTM
        lstm_output, _ = self.encoder(obs)

        # Take the last timestep output
        features = lstm_output[:, -1, :]

        # Get action logits
        action_logits = self.pi(features)

        return {"action_dist_inputs": action_logits}

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for exploration (same as inference for PPO)."""
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for training (keep embeddings for value func. call)."""
        obs = batch["obs"]

        # Flatten observation if needed and add sequence dimension
        if len(obs.shape) == 2:
            # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)

        # Forward pass through the LSTM
        lstm_output, _ = self.encoder(obs)

        # Take the last timestep output
        features = lstm_output[:, -1, :]

        # Get action logits
        action_logits = self.pi(features)

        # Return embeddings for value function computation
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.EMBEDDINGS: features,  # Keep embeddings for compute_values
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Any = None,
    ) -> TensorType:
        """Compute value function predictions for the given batch."""
        if embeddings is None:
            obs = batch["obs"]

            # Flatten observation if needed and add sequence dimension
            if len(obs.shape) == 2:
                # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
                obs = obs.unsqueeze(1)

            # Forward pass through the LSTM
            lstm_output, _ = self.encoder(obs)

            # Take the last timestep output
            embeddings = lstm_output[:, -1, :]

        # Value head
        vf_out = self.vf(embeddings)
        # Squeeze out last dimension (single node value head)
        return vf_out.squeeze(-1)
