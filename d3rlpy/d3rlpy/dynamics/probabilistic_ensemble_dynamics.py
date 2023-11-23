from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from ..iterators import RandomIterator, RoundIterator, TransitionIterator
from ..logger import LOG, D3RLPyLogger
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from .base import DynamicsBase
from .torch.probabilistic_ensemble_dynamics_impl import (
    ProbabilisticEnsembleDynamicsImpl,
)


class ProbabilisticEnsembleDynamics(DynamicsBase):
    r"""Probabilistic ensemble dynamics.

    The ensemble dynamics model consists of :math:`N` probablistic models
    :math:`\{T_{\theta_i}\}_{i=1}^N`.
    At each epoch, new transitions are generated via randomly picked dynamics
    model :math:`T_\theta`.

    .. math::

        s_{t+1}, r_{t+1} \sim T_\theta(s_t, a_t)

    where :math:`s_t \sim D` for the first step, otherwise :math:`s_t` is the
    previous generated observation, and :math:`a_t \sim \pi(\cdot|s_t)`.

    Note:
        Currently, ``ProbabilisticEnsembleDynamics`` only supports vector
        observations.

    References:
        * `Yu et al., MOPO: Model-based Offline Policy Optimization.
          <https://arxiv.org/abs/2005.13239>`_

    Args:
        learning_rate (float): learning rate for dynamics model.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_ensembles (int): the number of dynamics model for ensemble.
        variance_type (str): variance calculation type. The available options
            are ``['max', 'data']``.
        discrete_action (bool): flag to take discrete actions.
        scaler (d3rlpy.preprocessing.scalers.Scaler or str): preprocessor.
            The available options are ``['pixel', 'min_max', 'standard']``.
        action_scaler (d3rlpy.preprocessing.Actionscalers or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        impl (d3rlpy.dynamics.torch.ProbabilisticEnsembleDynamicsImpl):
            dynamics implementation.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _n_ensembles: int
    _variance_type: str
    _discrete_action: bool
    _use_gpu: Optional[Device]
    _impl: Optional[ProbabilisticEnsembleDynamicsImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(weight_decay=1e-4),
        state_encoder_factory: EncoderArg = "default",
        reward_encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        n_ensembles: int = 5,
        variance_type: str = "max",
        discrete_action: bool = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        use_gpu: UseGPUArg = False,
        impl: Optional[ProbabilisticEnsembleDynamicsImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._state_encoder_factory = check_encoder(state_encoder_factory)
        self._reward_encoder_factory = check_encoder(reward_encoder_factory)
        self._n_ensembles = n_ensembles
        self._variance_type = variance_type
        self._discrete_action = discrete_action
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = ProbabilisticEnsembleDynamicsImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            state_encoder_factory=self._state_encoder_factory,
            reward_encoder_factory=self._reward_encoder_factory,
            n_ensembles=self._n_ensembles,
            variance_type=self._variance_type,
            discrete_action=self._discrete_action,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch, permutation_matrices=None) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch, permutation_matrices[0], permutation_matrices[1])
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.BOTH

    def fit(
            self,
            dataset: Union[List[Episode], List[Transition], MDPDataset],
            n_epochs: Optional[int] = None,
            n_steps: Optional[int] = None,
            n_steps_per_epoch: int = 10000,
            save_metrics: bool = True,
            experiment_name: Optional[str] = None,
            with_timestamp: bool = True,
            logdir: str = "d3rlpy_logs",
            verbose: bool = True,
            show_progress: bool = True,
            tensorboard_dir: Optional[str] = None,
            eval_episodes: Optional[List[Episode]] = None,
            save_interval: int = 1,
            scorers: Optional[
                Dict[str, Callable[[Any, List[Episode]], float]]
            ] = None,
            shuffle: bool = True,
            callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
            permutation_matrices = None,
        ) -> List[Tuple[int, Dict[str, float]]]:
            """Trains with the given dataset.

            .. code-block:: python

                algo.fit(episodes, n_steps=1000000)

            Args:
                dataset: list of episodes to train.
                n_epochs: the number of epochs to train.
                n_steps: the number of steps to train.
                n_steps_per_epoch: the number of steps per epoch. This value will
                    be ignored when ``n_steps`` is ``None``.
                save_metrics: flag to record metrics in files. If False,
                    the log directory is not created and the model parameters are
                    not saved during training.
                experiment_name: experiment name for logging. If not passed,
                    the directory name will be `{class name}_{timestamp}`.
                with_timestamp: flag to add timestamp string to the last of
                    directory name.
                logdir: root directory name to save logs.
                verbose: flag to show logged information on stdout.
                show_progress: flag to show progress bar for iterations.
                tensorboard_dir: directory to save logged information in
                    tensorboard (additional to the csv data).  if ``None``, the
                    directory will not be created.
                eval_episodes: list of episodes to test.
                save_interval: interval to save parameters.
                scorers: list of scorer functions used with `eval_episodes`.
                shuffle: flag to shuffle transitions on each epoch.
                callback: callable function that takes ``(algo, epoch, total_step)``
                    , which is called every step.
                permutation_matrices: (P_s, P_a) that permutates data

            Returns:
                list of result tuples (epoch, metrics) per epoch.

            """
            results = list(
                self.fitter(
                    dataset,
                    n_epochs,
                    n_steps,
                    n_steps_per_epoch,
                    save_metrics,
                    experiment_name,
                    with_timestamp,
                    logdir,
                    verbose,
                    show_progress,
                    tensorboard_dir,
                    eval_episodes,
                    save_interval,
                    scorers,
                    shuffle,
                    callback,
                    permutation_matrices = permutation_matrices,
                )
            )
            return results

    def fitter(
            self,
            dataset: Union[List[Episode], List[Transition], MDPDataset],
            n_epochs: Optional[int] = None,
            n_steps: Optional[int] = None,
            n_steps_per_epoch: int = 10000,
            save_metrics: bool = True,
            experiment_name: Optional[str] = None,
            with_timestamp: bool = True,
            logdir: str = "d3rlpy_logs",
            verbose: bool = True,
            show_progress: bool = True,
            tensorboard_dir: Optional[str] = None,
            eval_episodes: Optional[List[Episode]] = None,
            save_interval: int = 1,
            scorers: Optional[
                Dict[str, Callable[[Any, List[Episode]], float]]
            ] = None,
            shuffle: bool = True,
            callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
            permutation_matrices = None,
        ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
            """Iterate over epochs steps to train with the given dataset. At each
                iteration algo methods and properties can be changed or queried.

            .. code-block:: python

                for epoch, metrics in algo.fitter(episodes):
                    my_plot(metrics)
                    algo.save_model(my_path)

            Args:
                dataset: offline dataset to train.
                n_epochs: the number of epochs to train.
                n_steps: the number of steps to train.
                n_steps_per_epoch: the number of steps per epoch. This value will
                    be ignored when ``n_steps`` is ``None``.
                save_metrics: flag to record metrics in files. If False,
                    the log directory is not created and the model parameters are
                    not saved during training.
                experiment_name: experiment name for logging. If not passed,
                    the directory name will be `{class name}_{timestamp}`.
                with_timestamp: flag to add timestamp string to the last of
                    directory name.
                logdir: root directory name to save logs.
                verbose: flag to show logged information on stdout.
                show_progress: flag to show progress bar for iterations.
                tensorboard_dir: directory to save logged information in
                    tensorboard (additional to the csv data).  if ``None``, the
                    directory will not be created.
                eval_episodes: list of episodes to test.
                save_interval: interval to save parameters.
                scorers: list of scorer functions used with `eval_episodes`.
                shuffle: flag to shuffle transitions on each epoch.
                callback: callable function that takes ``(algo, epoch, total_step)``
                    , which is called every step.

            Returns:
                iterator yielding current epoch and metrics dict.

            """

            transitions = []
            if isinstance(dataset, MDPDataset):
                for episode in dataset.episodes:
                    transitions += episode.transitions
            elif not dataset:
                raise ValueError("empty dataset is not supported.")
            elif isinstance(dataset[0], Episode):
                for episode in cast(List[Episode], dataset):
                    transitions += episode.transitions
            elif isinstance(dataset[0], Transition):
                transitions = list(cast(List[Transition], dataset))
            else:
                raise ValueError(f"invalid dataset type: {type(dataset)}")

            # check action space
            if self.get_action_type() == ActionSpace.BOTH:
                pass
            elif transitions[0].is_discrete:
                assert (
                    self.get_action_type() == ActionSpace.DISCRETE
                ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
            else:
                assert (
                    self.get_action_type() == ActionSpace.CONTINUOUS
                ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

            iterator: TransitionIterator
            if n_epochs is None and n_steps is not None:
                assert n_steps >= n_steps_per_epoch
                n_epochs = n_steps // n_steps_per_epoch
                iterator = RandomIterator(
                    transitions,
                    n_steps_per_epoch,
                    batch_size=self._batch_size,
                    n_steps=self._n_steps,
                    gamma=self._gamma,
                    n_frames=self._n_frames,
                    real_ratio=self._real_ratio,
                    generated_maxlen=self._generated_maxlen,
                )
                LOG.debug("RandomIterator is selected.")
            elif n_epochs is not None and n_steps is None:
                iterator = RoundIterator(
                    transitions,
                    batch_size=self._batch_size,
                    n_steps=self._n_steps,
                    gamma=self._gamma,
                    n_frames=self._n_frames,
                    real_ratio=self._real_ratio,
                    generated_maxlen=self._generated_maxlen,
                    shuffle=shuffle,
                )
                LOG.debug("RoundIterator is selected.")
            else:
                raise ValueError("Either of n_epochs or n_steps must be given.")

            # setup logger
            logger = self._prepare_logger(
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                tensorboard_dir,
            )

            # add reference to active logger to algo class during fit
            self._active_logger = logger

            # initialize scaler
            if self._scaler:
                LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
                self._scaler.fit(transitions)

            # initialize action scaler
            if self._action_scaler:
                LOG.debug(
                    "Fitting action scaler...",
                    action_scaler=self._action_scaler.get_type(),
                )
                self._action_scaler.fit(transitions)

            # initialize reward scaler
            if self._reward_scaler:
                LOG.debug(
                    "Fitting reward scaler...",
                    reward_scaler=self._reward_scaler.get_type(),
                )
                self._reward_scaler.fit(transitions)

            # instantiate implementation
            if self._impl is None:
                LOG.debug("Building models...")
                transition = iterator.transitions[0]
                action_size = transition.get_action_size()
                observation_shape = tuple(transition.get_observation_shape())
                self.create_impl(
                    self._process_observation_shape(observation_shape), action_size
                )
                LOG.debug("Models have been built.")
            else:
                LOG.warning("Skip building models since they're already built.")

            # save hyperparameters
            self.save_params(logger)

            # refresh evaluation metrics
            self._eval_results = defaultdict(list)

            # refresh loss history
            self._loss_history = defaultdict(list)

            # training loop
            total_step = 0
            for epoch in range(1, n_epochs + 1):

                # dict to add incremental mean losses to epoch
                epoch_loss = defaultdict(list)

                range_gen = tqdm(
                    range(len(iterator)),
                    disable=not show_progress,
                    desc=f"Epoch {int(epoch)}/{n_epochs}",
                )

                iterator.reset()

                for itr in range_gen:

                    # generate new transitions with dynamics models
                    new_transitions = self.generate_new_data(
                        transitions=iterator.transitions,
                    )
                    if new_transitions:
                        iterator.add_generated_transitions(new_transitions)
                        LOG.debug(
                            f"{len(new_transitions)} transitions are generated.",
                            real_transitions=len(iterator.transitions),
                            fake_transitions=len(iterator.generated_transitions),
                        )

                    with logger.measure_time("step"):
                        # pick transitions
                        with logger.measure_time("sample_batch"):
                            batch = next(iterator)

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            #loss = self.update(batch)
                            loss = self.update(batch, permutation_matrices)

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)
                            epoch_loss[name].append(val)

                        # update progress postfix with losses
                        if itr % 10 == 0:
                            mean_loss = {
                                k: np.mean(v) for k, v in epoch_loss.items()
                            }
                            range_gen.set_postfix(mean_loss)

                    total_step += 1

                    # call callback if given
                    if callback:
                        callback(self, epoch, total_step)

                # save loss to loss history dict
                self._loss_history["epoch"].append(epoch)
                self._loss_history["step"].append(total_step)
                for name, vals in epoch_loss.items():
                    if vals:
                        self._loss_history[name].append(np.mean(vals))

                if scorers and eval_episodes:
                    self._evaluate(eval_episodes, scorers, logger)

                # save metrics
                metrics = logger.commit(epoch, total_step)

                # save model parameters
                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                yield epoch, metrics

            # drop reference to active logger since out of fit there is no active
            # logger
            self._active_logger.close()
            self._active_logger = None
    def update(self, batch: TransitionMiniBatch, permutation_matrices=None) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        loss = self._update(batch, permutation_matrices)
        self._grad_step += 1
        return loss