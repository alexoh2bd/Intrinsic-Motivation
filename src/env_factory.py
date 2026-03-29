"""
Environment factory for scaling-CRL.

Centralizes env construction so train.py, train_shallow_actor.py, etc.
all share the same make_env logic.
"""


def make_env(env_id, args):
    """
    Create a Brax environment and set obs_dim / goal indices on args.

    Args:
        env_id: environment identifier string
        args: Args dataclass (will be mutated to set obs_dim, goal_start_idx, goal_end_idx)

    Returns:
        A Brax environment instance (unwrapped).
    """
    print(f"making env with env_id: {env_id}", flush=True)

    if env_id == "reacher":
        from envs.reacher import Reacher
        env = Reacher(backend="spring")
        args.obs_dim = 10
        args.goal_start_idx = 4
        args.goal_end_idx = 7

    elif env_id == "pusher":
        from envs.pusher import Pusher
        env = Pusher(backend="spring")
        args.obs_dim = 20
        args.goal_start_idx = 10
        args.goal_end_idx = 13

    elif env_id == "ant":
        from envs.ant import Ant
        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    elif "ant" in env_id and "maze" in env_id:
        if "gen" not in env_id:
            from envs.ant_maze import AntMaze
            env = AntMaze(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
                maze_layout_name=env_id[4:],
            )
            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2
        else:
            from envs.ant_maze_generalization import AntMazeGeneralization
            gen_idx = env_id.find("gen")
            maze_layout_name = env_id[4:gen_idx - 1]
            generalization_config = env_id[gen_idx + 4:]
            print(f"maze_layout_name: {maze_layout_name}, generalization_config: {generalization_config}", flush=True)
            env = AntMazeGeneralization(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
                maze_layout_name=maze_layout_name,
                generalization_config=generalization_config,
            )
            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2

    elif env_id == "ant_ball":
        from envs.ant_ball import AntBall
        env = AntBall(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        args.obs_dim = 31
        args.goal_start_idx = 28
        args.goal_end_idx = 30

    elif env_id == "ant_push":
        from envs.ant_push import AntPush
        env = AntPush(backend="mjx")
        args.obs_dim = 31
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    elif env_id == "humanoid":
        from envs.humanoid import Humanoid
        env = Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        args.obs_dim = 268
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze
        env = HumanoidMaze(
            backend="spring",
            maze_layout_name=env_id[9:],
        )
        args.obs_dim = 268
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach
        env = ArmReach(backend="mjx")
        args.obs_dim = 13
        args.goal_start_idx = 7
        args.goal_end_idx = 10

    elif env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
        env = ArmBinpickEasy(backend="mjx")
        args.obs_dim = 17
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard
        env = ArmBinpickHard(backend="mjx")
        args.obs_dim = 17
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif env_id == "arm_binpick_easy_EEF":
        from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
        env = ArmBinpickEasyEEF(backend="mjx")
        args.obs_dim = 11
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif "arm_grasp" in env_id:
        from envs.manipulation.arm_grasp import ArmGrasp
        cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
        env = ArmGrasp(cube_noise_scale=cube_noise_scale, backend="mjx")
        args.obs_dim = 23
        args.goal_start_idx = 16
        args.goal_end_idx = 23

    elif env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy
        env = ArmPushEasy(backend="mjx")
        args.obs_dim = 17
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    elif env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard
        env = ArmPushHard(backend="mjx")
        args.obs_dim = 17
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")

    return env
