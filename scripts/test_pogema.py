# test_pogema_debug.py
import pogema
import numpy as np
from pogema import GridConfig, pogema_v0

def test_debug():
    grid_config = GridConfig(
        num_agents=2,
        size=5,
        density=0.1,
        seed=42,
        max_episode_steps=200,
        obs_radius=2
    )

    env = pogema_v0(grid_config=grid_config)
    obs = env.reset()
    print("After reset — obs:", obs)
    print("Agent positions:", env.agent_positions if hasattr(env, "agent_positions") else None)
    print("Goal positions:", env.goals if hasattr(env, "goals") else None)

    done = False
    total_rew = [0, 0]
    step_count = 0
    while not done:
        action_list = [env.action_space.sample() for _ in range(grid_config.num_agents)]
        print(f"Step {step_count}, action_list:", action_list)

        ret = env.step(action_list)
        print(f"Step {step_count}, raw env.step() return:", ret)

        # Unpack according to correct API (Gym vs Gymnasium)
        if len(ret) == 4:
            obs, rew, done, info = ret
            terminated = truncated = done
        elif len(ret) == 5:
            obs, rew, terminated, truncated, info = ret
            done = all(terminated) or all(truncated)
        else:
            raise ValueError("Unexpected return signature for env.step()")

        print(f"Step {step_count}, rew:", rew)
        print(f"Step {step_count}, terminated:", terminated, "truncated:", truncated)
        total_rew = [tr + r for tr, r in zip(total_rew, rew)]
        step_count += 1

    print("Loop exited after steps:", step_count)
    print("Total reward per agent:", total_rew)

if __name__ == "__main__":
    print("Debugging POGEMA loop…")
    test_debug()
