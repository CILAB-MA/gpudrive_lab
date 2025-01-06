from pygpudrive.env.config import DynamicsModel, ActionSpace
from pygpudrive.env.env_torch import GPUDriveDiscreteEnv, GPUDriveMultiDiscreteEnv, GPUDriveContinuousEnv

def make(dynamics_id=None, action_space=None, kwargs=None) -> None:
    """
    Creates an environment with a specified dynamics model and action space.

    Args:
        dynamics_id (str, optional): Identifier for the dynamics model. Default is None.
        action_space (gym.Space, optional): The action space for the environment. Default is None.
        kwargs: config, scene_config, max_cont_agents, device, num_stack, render_config
    """
    match dynamics_id:
        case DynamicsModel.CLASSIC:
            match action_space:
                case ActionSpace.DISCRETE:
                    pass
                case ActionSpace.MULTI_DISCRETE:
                    pass
                case ActionSpace.CONTINUOUS:
                    pass
                case _:
                    raise NotImplementedError
            pass
        case DynamicsModel.BICYCLE:
            match action_space:
                case ActionSpace.DISCRETE:
                    pass
                case ActionSpace.MULTI_DISCRETE:
                    pass
                case ActionSpace.CONTINUOUS:
                    pass
                case _:
                    raise NotImplementedError
        case DynamicsModel.DELTA_LOCAL:
            match action_space:
                case ActionSpace.DISCRETE:
                    return GPUDriveDiscreteEnv(**kwargs)
                case ActionSpace.MULTI_DISCRETE:
                    return GPUDriveMultiDiscreteEnv(**kwargs)
                case ActionSpace.CONTINUOUS:
                    return GPUDriveContinuousEnv(**kwargs)
                case _:
                    raise NotImplementedError
        case DynamicsModel.STATE:
            match action_space:
                case ActionSpace.DISCRETE:
                    pass
                case ActionSpace.MULTI_DISCRETE:
                    pass
                case ActionSpace.CONTINUOUS:
                    pass
                case _:
                    raise NotImplementedError
        case _:
            raise NotImplementedError    
        
    