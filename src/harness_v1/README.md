# Harness v1 | L3-04 System 2

This directory contains the System 2 MCP client and the 60 motion requests recorded during one interactive LLM4Robot-Bench L3-04 sweep (seed 0, robot gravity compensation enabled). The model read RGB-D observations and tool feedback between moves. It sent world-frame end-effector positions with WXYZ quaternions, or gripper commands. The simulator and cuRobo planned and executed each move. Brush pickup, visual recalibration, and raising the brush near the dustpan entrance drew on earlier Code-as-Policy (CaP) trials.

The recorded pilot had 51 executed actions, nine planner rejections with zero execution steps, and zero controller timeouts. The native evaluator counted **2 of 3 blocks**. The last sweep placed the right-rear block inside the pan. An earlier lift displaced the left-rear block so that two corners extended beyond the scored floor boundary. This one development run used prior experience with the same layout.

## Requirements

- Python 3.10 or later for the client. It uses the standard library.
- A running **simulation** MCP endpoint with a compatible R5 adapter and the ASPIRE System 2 bridge. The benchmark checkout and this client alone do not provide that server.
- For the recorded setup, the server used the L3-04 scene/assets, Isaac Sim 5.1, a compatible NVIDIA GPU, cuRobo, and [ASPIRE](https://github.com/NVlabs/ASPIRE/tree/fa00f0ef2bc1d91d594b9aa502c22133c9f538f7)'s embedded runtime and MCP bridge.
- Before replay, reset the server to L3-04 with seed 0 and gravity compensation enabled. The task tool verifies the task ID. Confirm seed and gravity compensation in the server's run configuration.

## Interactive session

Set `MCP_URL` to the simulation endpoint. These are example calls against an already running service.

```bash
MCP_URL=http://127.0.0.1:8765/mcp/
python src/harness_v1/system2_client.py --url "$MCP_URL" list
python src/harness_v1/system2_client.py --url "$MCP_URL" call r5_task '{}'
python src/harness_v1/system2_client.py --url "$MCP_URL" \
  call observe '{"cameras":["overview","left_wrist"]}' --images observations
```

Use `r5_pixel_to_world` to query depth at native image coordinates. The model or operator can issue one `cap_move_to_pose` or `cap_set_gripper` request for the public `left` manipulation arm and inspect the terminal job and new observation. If the planner refuses a move before execution, use the returned error and observation to choose the next target. A transport failure may leave the move in flight. Inspect the job or current state before issuing another motion.

## Replay the recorded pilot

```bash
python src/harness_v1/system2_client.py --url "$MCP_URL" replay
```

This sends the recorded actions in order, waits for each result, and prints one JSON record per move. It stops if a status differs from the recording. The recorded planner rejections are included for fidelity. Replay uses recorded commands without a model API call. Contact dynamics may differ, so evaluate the resulting episode with the benchmark's native scorer.

- `system2_client.py` provides the interactive tool calls and sequential replay.
- `l3_04_seed0_actions.json` records requests, terminal statuses, and executed steps. It contains no images or server audit state.
