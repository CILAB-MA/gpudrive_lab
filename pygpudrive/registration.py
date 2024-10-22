from pygpudrive.env.config import DynamicsModel, ActionSpace
from pygpudrive.env.env_torch import GPUDriveDiscreteEnv, GPUDriveMultiDiscreteEnv, GPUDriveContinuousEnv

def make(dynamics_id=None, action_id=None, kwargs=None) -> None:
    """Creates a environment with dynamics model and action space specified."""
    match dynamics_id:
        case DynamicsModel.CLASSIC:
            match action_id:
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
            match action_id:
                case ActionSpace.DISCRETE:
                    pass
                case ActionSpace.MULTI_DISCRETE:
                    pass
                case ActionSpace.CONTINUOUS:
                    pass
                case _:
                    raise NotImplementedError
        case DynamicsModel.DELTA_LOCAL:
            match action_id:
                case ActionSpace.DISCRETE:
                    return GPUDriveDiscreteEnv(**kwargs)
                case ActionSpace.MULTI_DISCRETE:
                    return GPUDriveMultiDiscreteEnv(**kwargs)
                case ActionSpace.CONTINUOUS:
                    return GPUDriveContinuousEnv(**kwargs)
                case _:
                    raise NotImplementedError
        case DynamicsModel.STATE:
            match action_id:
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
        
    