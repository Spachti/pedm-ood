from typing import Optional
from datetime import datetime

import numpy as np
import gym

from ood_baselines.common.base_detector import Base_Detector

from pedm.nn_models.default_cfg import model_cfg_dict
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.algorithm.dreamer import Dreamer

from torch.utils.tensorboard import SummaryWriter

class Dreamer_Detector(Base_Detector):
    """Dreamer Dynamics Model ODD Detector for RL agents"""

    def __init__(
        self,
        env,
        dyn_model_kwargs={},
        n_part: Optional[int] = 1_000,
        horizon: Optional[int] = 1,
        criterion: Optional[str] = "pred_error_samples",
        aggregation_function: Optional[str] = "min_mean",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            dyn_model: (probabilistic) dynamics model for the detector
            n_part: number of particles to sample from the detector
            criterion: specifies method to calculate anomaly score from model predictions
            aggregation_function: specifies method to aggregate anomaly scores for multiple particles
            normalize_data: flag to normalize all data ((X-mean)/std)
        """

        if not dyn_model_kwargs:
            dyn_model_kwargs = model_cfg_dict[env.spec.id]["dyn_model_kwargs"]

        self.env = env
        obs_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            discrete_action_bool = True
            action_size = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            discrete_action_bool = False
            action_size = env.action_space.shape[0]
        
        config = load_config("dmc-reacher-easy.yml")

        log_dir = (
            get_base_directory()
            + "/runs/"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + config.operation.log_dir
        )
        writer = SummaryWriter(log_dir)
        self.dyn_model = Dreamer(obs_shape, discrete_action_bool, action_size, writer, "cuda", config)
        self.n_part = n_part
        self.criterion = criterion
        self.aggregation_function = aggregation_function
        if horizon != 1:
            raise NotImplementedError
        super().__init__(*args, **kwargs)

    
    def _fit(self, *args, **kwargs):
        return self.dyn_model.train(self.env)
    
    def _predict_scores(self, obs, acts) -> np.ndarray:
        preds = []
        posterior, determinisitc = self.dyn_model.rssm.recurrent_model_input_init(1)
        for observation, action in zip(obs[:-1], acts):
            determinisitc = self.dyn_model.rssm.recurrent_model(
                posterior, action, determinisitc
            )
            embedded_observation = self.dyn_model.encoder(observation).reshape(1, -1)
            _, posterior = self.dyn_model.rssm.representation_model(
                embedded_observation, determinisitc
            )
            next_observation = self.dyn_model.decoder(posterior, determinisitc).sample()
            print(next_observation)
            preds.append(next_observation)

        return super()._predict_scores(obs, acts)