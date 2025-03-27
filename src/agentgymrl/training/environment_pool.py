import concurrent.futures
from dataclasses import dataclass, field
import logging
from typing import Generic, Optional, List, Dict, Any, Type, TypeVar

from agentgymrl.environments.environment import Environment, EnvironmentResult
from agentgymrl.environments.state import STATE
from agentgymrl.inference.model_output import ModelOutput


T = TypeVar("T", bound=Environment)


@dataclass
class EnvironmentConfig(Generic[T]):
    """
    This dataclass encapsulates all parameters needed to create and manage environments in parallel.

    Example:
        ```
        config = EnvironmentConfig(
            env_class=MyEnvironment,
            num_environments=2,
            env_class_shared_kwargs={"shared_arg": "foo"},
            env_class_individual_kwargs=[
                {"port_num": "8000"},
                {"port_num": "8001"},
            ]
        )
        pool = EnvironmentPool(config)
        ```
    """

    env_class: Type[T]
    num_envs: int
    env_class_shared_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    env_class_individual_kwargs: Optional[List[Dict[str, Any]]] = field(
        default_factory=list
    )

    def __post_init__(self):
        """
        Validate the configuration after initialization.
        """
        if self.env_class_shared_kwargs is None:
            self.env_class_shared_kwargs = {}

        # Ensure individual_kwargs is a list of the right length
        if (
            self.env_class_individual_kwargs is None
            or len(self.env_class_individual_kwargs) == 0
        ):
            self.env_class_individual_kwargs = [{} for _ in range(self.num_envs)]
        elif len(self.env_class_individual_kwargs) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} individual kwargs dictionaries, "
                f"got {len(self.env_class_individual_kwargs)}"
            )


class EnvironmentPool(Generic[STATE]):
    """
    This class is responsible for creating and cleaning up environments,
    as well as distributing model outputs to the correct environment.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
    ):
        """
        Initialize a pool of environments.

        Args:
            config: EnvironmentConfig object containing environment creation parameters
        ```
        """
        self.logger = logging.getLogger(__name__)
        self.env_class = config.env_class
        self.num_environments = config.num_environments
        self.environments: List[T] = []


        if (
            config.env_class_individual_kwargs is None
            or len(config.env_class_individual_kwargs) == 0
        ):
            env_class_individual_kwargs = [{} for _ in range(config.num_environments)]
        elif len(env_class_individual_kwargs) != config.num_environments:
            raise ValueError(
                f"Expected {config.num_environments} individual kwargs dictionaries, got {len(env_class_individual_kwargs)}"
            )

        self._create_environments_parallel(
            config.env_class_shared_kwargs or {}, env_class_individual_kwargs
        )

    def _create_environment(self, idx: int, env_args: dict) -> T:
        """
        Create a single environment with the given parameters.

        Args:
            idx: Index of the environment to create
            env_args: Dictionary of kwargs to pass to the environment class

        Returns:
            Created environment instance

        Raises:
            Exception: If environment creation fails
        """
        try:
            env = self.env_class(**env_args, env_idx=idx)
            self.logger.debug(f"Initialized environment {idx} with kwargs: {env_args}")
            return env
        except Exception as e:
            self.logger.error(f"Failed to initialize environment {idx}: {e}")
            raise

    def _create_environments_parallel(
        self,
        env_class_shared_kwargs: Dict[str, Any],
        env_class_individual_kwargs: List[Dict[str, Any]],
    ) -> None:
        """
        Create environments in parallel using ThreadPoolExecutor.

        Args:
            env_class_shared_kwargs: Dictionary of kwargs to pass to all instantiations of the env_class
            env_class_individual_kwargs: List of dictionaries of kwargs to pass to each instantiation of the env_class.

        Raises:
            EnvironmentError: If any environment fails to initialize
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._create_environment,
                    i,
                    env_args={
                        **env_class_shared_kwargs,
                        **env_class_individual_kwargs[i],
                    },
                )
                for i in range(self.num_environments)
            ]

            failed_envs = []

            for i, future in enumerate(futures):
                try:
                    res = future.result()
                    self.environments.append(res)
                except Exception as e:
                    failed_envs.append((i, str(e)))

            if failed_envs:
                self.cleanup()
                error_msg = "\n".join(
                    [f"Environment {i}: {err}" for i, err in failed_envs]
                )
                raise EnvironmentError(
                    f"Failed to initialize {len(failed_envs)} environments:\n{error_msg}"
                )

            self.logger.info(
                f"Successfully created {len(self.environments)} environments in parallel"
            )

    def handle_output(
        self, env_idx: int, model_output: ModelOutput
    ) -> EnvironmentResult:
        """
        Execute a tool in a specific environment.

        Args:
            env_idx: Index of the environment to use
            model_output: Output from the model

        Returns:
            EnvironmentResult from the environment

        Raises:
            IndexError: If env_idx is out of range
        """
        if env_idx < 0 or env_idx >= len(self.environments):
            raise IndexError(
                f"Environment index {env_idx} out of range (0-{len(self.environments) - 1})"
            )

        return self.environments[env_idx].handle_output(model_output)
    
    def get_state(self, env_idx: int) -> Optional[STATE]:
        """
        Get the state of a specific environment.

        Args:
            env_idx: Index of the environment to use

        Returns:
            State dictionary of the environment

        Raises:
            IndexError: If env_idx is out of range
        """
        if env_idx < 0 or env_idx >= len(self.environments):
            raise IndexError(
                f"Environment index {env_idx} out of range (0-{len(self.environments) - 1})"
            )

        return self.environments[env_idx].get_state()

    def cleanup(self) -> None:
        """Clean up resources used by all environments."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(env.cleanup) for env in self.environments]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up environment: {e}")

    def __enter__(self) -> "EnvironmentPool[T]":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and clean up resources."""
        self.cleanup()
