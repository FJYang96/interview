## "Learning-based legged locomotion: state of the art and future perspectives", Ha et al, IJRR 2025

### Introduction
- The current surge of quadrupedal and bipedal robots benefitted from a convergence to the so-called "proprioceptive actuation"
    - previously, electric motors-based actuation require high-ratio gears that have tremendous internal friction
    - This is this idea that we use high torque-density motors and thereby avoid high-ratio gears
- Simulators have developped tremendously in the same time
    - fast inverse dynamics can now be solved in milliseconds using packages like Pinoccio
    - stiff, complementarity-based rigid contact models are now available and are GPU parallelizable
- Control algorithms have went through revolutions from heuristics -> CPG (cyclic/periodic) -> MPC -> RL
    - MPC-based approach have achieved tremendous success, but dealing with uncertainty in contact and incorporating high-dimensional sensory modalities is challenging.

### Components of the Locomotion MDP
- modern simulators often use *rigid* contact models:
    - *rigid* means that no deformation of the object is allowed, and is opposite to *compliant*
    - rigid contact can be elastic or inelastic. Inelastic means the normal velocity becomes 0 at contact (no bouncing)
- Observations typically include:
    - *proprioception* (info on the internal state of the robot):
        - include sensors like IMU, joint encoders, contact sensors that are often noisy
        - usually filtered with a state estimator to get base pose, base twist, joint psotion, and velocity
        - keeping a history of such information as observation is tremendously helpful
        - keeping a history of _actions_ can also be very beneficial as info used for sysID
    - *exteroception* (info on the external environment around the robot):
        - early works use elevation map around the robot as part of the observation
        - recent works directly use raw sensory readings like depth images or point clouds
        - rgb images are sometimes used for scene understanding
    - *task related inputs* (goals):
        - velocity/pose commands
        - planned end-effector trajectory (so that we can achieve whole-body motions)
- Rewards (commonly linear combination of terms, sometimes also multiplicative)
    - *rule of thumb*:
        - we typically want the rewards to be bounded (by either clipping or applying the exp kernel, e.g. $$\exp(-c||e||)$$)
    - *common reward terms include*:
        - horizontal velocity error, yaw rate error, base vertical vel, roll and pitch, z-axis deviation, joint velocities, joint accelerations, joint torques, joint mechanical power, action rate, action smoothness
    - *imitation reward*
- Actions:
    - most works use joint target position as the action and wrap a PD controller underneath (usually called PD policy)
    - torque policies give more fine-grained control, but struggle to be run at high-enough frequency
    - other high-level actions in the task space (e.g. foot position) are also possible
    - one can additionally run a filter 
    - actual torque commands are usually generated at around 1kHz
    - PD policies can run at ~50Hz and still achieve decent performance


### Learning Tricks
- Curriculum learning:
    - one difficulty is in deciding when to move on to the next stage of the curriculum
- Priviledged learning:
    - train a teacher policy with privileged sensing information and then distill onto a student


### Sim-to-Real
- *Design choices* to avoid overfitting to sim
    - rewards
        - avoid jittery motion -> penalize joint acceleration
        - avoid foot-dragging -> reward for foot air time
        - avoid stomping -> penalize foot impact
    - observation and action space
        - low proportional gain for the joint PD controller allows compliant behavior
            - $$k_p = 40$$ -> more like a torque controller
            - $$k_p = 160$$ -> more like a position controller
        - use state estimator to estimate body velocity as part of estimation
            - Xie et al found that such estimator can be biased from integrated accelerometer error (drifting while stepping in place); they add a constant drift to counter this
    - domain knowledge such as left-right symmetry, CPG, etc
- *System Identification*
    - a major source of error comes from the actuator dynamics
        - analytical model by identifying parameters (Tan et al., 2018)
        - black box learned model (because of lack of access to the motor internal states, try to predict a model from action to torque using a history of past joint positions and velocities)
    - another important source of the gap is in the rigid contact model
- *Domain randomization*
    - commonly randomized parameters include:
        - mass
        - friction coefficients
        - proportional gain (used somewhat as a proxy for motor internal friction)
        - latency
        - lateral disturbance forces
        - terrain elevation / slope
        - visual perception parameters (camera intrisics/extrinsics)
- *Domain adaptation*
    - during training, models are conditioned on environment parameters, which are then identified online
        - when trained successfully, adaptation-based policies can typically achieve better performance than randomization
    - early works explicitly identify environment parameters online: hard to extend to large number of env params
    - newer works implicitly represent the env parameters during the adaptation

### Loco-manipulation
- To read:
    - dr eureka
    - Deep WBC, Visual WBC for legged loco-manipulation, Cascaded compositional residual learning for complex interactive behaviors
    - humanplus, Sim-to-real learning for humanoid box loco-manipulation, Deep imitation learning for humanoid loco-manipulation through human teleoperation
    - omni-retarget, asap

## Rapid Motor Adaptation
- Priviledged policy
    - dynamics:
        - early termination if the robot CoM drops below a given height or roll/pitch too large
    - observations (30D)
        - joint position - 12D
        - joint velocity - 12D
        - torso roll and pitch - 2D -> (directly from the IMU)
        - foot contact - 4D
        - *extrisics vector (8D)*: compressed from an encoder module from a 17D extrinsics
    - action (12D): desired joint position (PD policy)
    - reward:
        - forward velocity (+), lateral movement/rotation (-), motor power (-), ground impact (-), smoothness (-), action magnitude (-), joint speed (-), orientation (-), z velocity (-), foot slip (-)
        - interestingly, the penalties are not clipped and 2 norm instead of exponentials are used
        - start training with very small penalty terms and exponentially anneal it up per iteration
- Adaptation module:
    - use the past 50 steps of state and actions to estimate the extrisics
    - instead of directly predicting the extrinsics, predict the encoded version $$z_t$$
    - to train the adaptation module, we will roll out the base policy with extrinsics predicted from the adaptation module instead of GT extrinsics
- Randomization:
    - friction, PD gains, payload, CoM offset, motor strength, re-sample probability

- Problems and fixes identified in future works:
    - priviledged info mismatch: there is a distribution shift between the GT extrinsics and its estimation from the adaptation module.
        - Soln: 