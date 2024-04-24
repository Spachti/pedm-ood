from gymnasium.envs.registration import register

register(id="MJCartpole-v0", entry_point="mujoco_envs.base_envs.cartpole:CartpoleEnv", max_episode_steps=200)

register(id="Pusher-v4", entry_point="mujoco_envs.base_envs.pusher:PusherEnv", max_episode_steps=150)

register(id="Reacher-v4", entry_point="mujoco_envs.base_envs.reacher:ReacherEnv", max_episode_steps=150)

register(id="MJHalfCheetah-v0", entry_point="mujoco_envs.base_envs.half_cheetah:HalfCheetahEnv", max_episode_steps=1000)

###############################
#          anom envs          #
###############################

register(id="AnomMJCartpole-v0", entry_point="mujoco_envs.mod_envs:AnomCartpoleEnv", max_episode_steps=200)

register(id="AnomMJHalfCheetah-v0", entry_point="mujoco_envs.mod_envs:AnomHalfCheetahEnv", max_episode_steps=1000)

register(id="AnomPusher-v4", entry_point="mujoco_envs.mod_envs:AnomPusherEnv", max_episode_steps=150)

register(id="AnomReacher-v4", entry_point="mujoco_envs.mod_envs:AnomReacherEnv", max_episode_steps=150)

###############################
#           mod envs          #
###############################

register(
    id="ModMJCartpole-v0",
    entry_point="mujoco_envs.mod_envs:ModCartpoleEnv",
    max_episode_steps=200,
)

register(
    id="ModMJHalfCheetah-v0",
    entry_point="mujoco_envs.mod_envs:ModHalfCheetahEnv",
    max_episode_steps=1000,
)

register(id="ModPusher-v4", entry_point="mujoco_envs.mod_envs:ModPusherEnv", max_episode_steps=150)

register(id="ModReacher-v4", entry_point="mujoco_envs.mod_envs:ModReacherEnv", max_episode_steps=150)
