# Derived from the X-VLA RoboTwin evaluation adapter.
# Copyright 2025 2toINF (https://github.com/2toINF)
# Licensed under the Apache License, Version 2.0; see LICENSE.
# Modified for XVLA-TCLA: joint actions, shared paths and unified policy client.
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from configuration import absolute_path, load_paths
from inference.client import PolicyClient


def read_yaml(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_env(task_name, task_config):
    env = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
    config_root = Path("task_config")
    args = read_yaml(config_root / f"{task_config}.yml")
    embodiments = read_yaml(config_root / "_embodiment_config.yml")
    cameras = read_yaml(config_root / "_camera_config.yml")
    embodiment = args["embodiment"]
    args["head_camera_h"] = cameras[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = cameras[args["camera"]["head_camera_type"]]["w"]
    if len(embodiment) == 1:
        left, right = embodiment[0], embodiment[0]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left, right = embodiment[:2]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("RoboTwin embodiment config must contain 1 or 3 entries")
    for side, name in (("left", left), ("right", right)):
        folder = embodiments[name]["file_path"]
        args[f"{side}_robot_file"] = folder
        args[f"{side}_embodiment_config"] = read_yaml(Path(folder) / "config.yml")
    args.update(
        task_name=task_name,
        task_config=task_config,
        embodiment_name="+".join(map(str, embodiment)),
        eval_mode=True,
        render_freq=0,
        policy_name="XVLA-TCLA",
        ckpt_setting="policy",
    )
    return env, args


def qpos_from_obs(obs):
    joint = obs["joint_action"]
    return np.concatenate(
        (
            joint["left_arm"],
            [joint["left_gripper"]],
            joint["right_arm"],
            [joint["right_gripper"]],
        )
    ).astype(np.float32)


def evaluate_task(
    client, task, task_config, episodes, seed, max_steps, output, save_video
):
    env, args = load_env(task, task_config)
    env.suc = env.test_num = 0
    completed = 0
    candidate_seed = 2000 * (seed + 1)
    attempts = 0
    camera_map = {
        "cam_high": "head_camera",
        "cam_left_wrist": "left_camera",
        "cam_right_wrist": "right_camera",
    }
    output.mkdir(parents=True, exist_ok=True)
    while completed < episodes:
        attempts += 1
        if attempts > episodes * 100:
            raise RuntimeError(f"Could not find enough valid expert seeds for {task}")
        selected_seed = candidate_seed
        candidate_seed += 1
        try:
            env.setup_demo(
                now_ep_num=completed, seed=selected_seed, is_test=True, **args
            )
            env.play_once()
            valid = env.plan_success and env.check_success()
        except Exception as exc:
            print(f"Expert seed {selected_seed} rejected: {exc}")
            valid = False
        finally:
            env.close_env()
        if not valid:
            continue
        success = False
        frames = []
        try:
            env.setup_demo(
                now_ep_num=completed, seed=selected_seed, is_test=True, **args
            )
            env.actor_pose = True
            client.reset(task)
            for _ in range(max_steps):
                obs = env.get_obs()
                images = [
                    obs["observation"][camera_map[name]]["rgb"]
                    for name in client.metadata["camera_names"]
                ]
                action = client.act(qpos_from_obs(obs), images)
                env.take_action(action, action_type="qpos")
                if save_video:
                    frames.append(env.get_obs()["observation"]["head_camera"]["rgb"])
                if env.check_success():
                    success = True
                    break
                if not env.actor_pose:
                    break
        finally:
            env.close_env()
        completed += 1
        env.suc += int(success)
        env.test_num = completed
        result = {
            "task": task,
            "episode": completed,
            "seed": selected_seed,
            "success": success,
        }
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")
        if frames:
            import imageio.v2 as imageio

            imageio.mimsave(output / f"{completed}_{int(success)}.mp4", frames, fps=30)
        print(f"{task}: {env.suc}/{completed}")
    result = {
        "task": task,
        "episodes": completed,
        "successes": env.suc,
        "success_rate": env.suc / completed,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate raw or latent XVLA-TCLA in RoboTwin 2.0."
    )
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--task", required=True, help="Task name or all (tasks from checkpoint)"
    )
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--output")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()
    paths = load_paths().paths
    output = Path(absolute_path(args.output) if args.output else paths.eval_output)
    root = Path(paths.robotwin)
    if not (root / "envs").is_dir():
        raise FileNotFoundError("Set paths.robotwin to your RoboTwin installation")
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    client = PolicyClient(args.server_url)
    if client.metadata["dataset"] != "robotwin":
        raise ValueError("The server must load a RoboTwin checkpoint")
    sys.path.insert(0, str(root))
    os.chdir(root)
    tasks = client.metadata["task_names"] if args.task == "all" else [args.task]
    try:
        for task in tasks:
            evaluate_task(
                client,
                task,
                args.task_config,
                args.episodes,
                args.seed,
                args.max_steps,
                output / task,
                args.save_video,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
