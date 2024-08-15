#pragma once

#include <madrona/math.hpp>
#include <madrona/types.hpp>
using namespace madrona;
using namespace madrona::math;
namespace gpudrive
{
    inline void forwardDeltaModel(DeltaAction &action, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // start delta model
            // start DeltaLocal
        const float dt{0.1};
        float yaw = utils::quatToYaw(rotation);
        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(yaw);
        float sin = std::sin(yaw);

        action.dx = action.dx * cos - action.dy * sin;
        action.dy = action.dx * sin + action.dy * cos;
            // end DeltaLocal
            // start DeltaGlobal
        position.x = position.x + action.dx;
        position.y = position.y + action.dy;

        velocity.linear.x = action.dx / dt;
        velocity.linear.y = action.dy / dt;
            // end DeltaGlobal
        float new_yaw = yaw + action.dyaw;

        // end delta model

    }

    inline DeltaAction inverseDeltaModel(const Rotation &rotation, const Position &position, const Rotation &targetRotation, const Position &targetPosition)
    {
        const float dt{0.1};

        DeltaAction action = {0, 0, 0};
        float yaw = utils::NormalizeAngle<float>(utils::quatToYaw(rotation));
        float target_yaw = utils::NormalizeAngle<float>(utils::quatToYaw(targetRotation));
        // start delta model
            // start DeltaGlobal
        action.dx = targetPosition.x - position.x;
        action.dy = targetPosition.y - position.y;
        action.dyaw = target_yaw - yaw;

        action.dx  = fmaxf(-6.0, fminf(action.dx, 6.0));
        action.dy = fmaxf(-6.0, fminf(action.dy, 6.0));

            // end DeltaGlobal
            // start DeltaLocal
        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(-yaw);
        float sin = std::sin(-yaw);

        float local_dx= action.dx * cos - action.dy * sin;
        float local_dy = action.dx * sin + action.dy * cos;

        action.dx = fmaxf(-6.0, fminf(local_dx, 6.0));
        action.dy = fmaxf(-6.0, fminf(local_dy, 6.0));
            // end DeltaLocal
        // end delta model
        
        return action;

    }

}