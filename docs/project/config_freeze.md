# SELMA — Config Freeze (19 July 2026)

## Versions
- lbforaging 2.0.0  (food count caps at 4f on 15x15; LLM-MCA's 15x15-*-5f NOT registered in this version)
- EPyMARL commit 6bb15ff (per-agent reward logging, validated)
- torch 2.12.1+cu130, Python 3.10.20, numpy 1.26.4, protobuf 3.20.3, tensorboard 2.15.2
- WSL2 Ubuntu, GTX 1650; pip check clean

## Candidate configs (both non-coop, both cooperative common-reward tasks)
- lbforaging:Foraging-15x15-4p-3f-v3   <- matches LLM-MCA's environment configuration; their QMIX 0.12+/-0.06
     (NOT a protocol replication: they used no common-reward/scalarisation, double-DQN)
- lbforaging:Foraging-15x15-4p-4f-v3   <- fallback; removes 4-agents/3-food scarcity confound

## Key settings
- common_reward: True, reward_scalarisation: sum  (=> single shared team reward for QMIX)
- force_coop: NOT used. Wording for writeup: "standard LBF configuration with
  team-level reward scalarisation, giving a fully cooperative common-reward task;
  LBF's force-cooperation mode is not used."
- env_args.time_limit = 50, t_max = 2_000_000
- QMIX defaults: lr 5e-4, gamma 0.99, batch 32, buffer 5000, target update 200,
  eps 1.0->0.05 over 50k, standardise_rewards True, test_nepisode 100, test_interval 50k
- Seeds: 1-5 for final runs; pilot/calibration seeds 91, 99 (kept disjoint)
- Five runs, as in SEAC and LLM-MCA (count matches; seed values are ours)

## Measured timing
- 200k steps = 19.5 min  =>  2M ~= 3.2 h/seed

## Environment notes
- tensorboard-logger segfaults in configure(); replaced with torch SummaryWriter (commit above)
- protobuf<3.21 pin retained (was NOT the segfault cause; not yet proven unnecessary)
- smaclite disabled (lines 6, 37 of src/envs/__init__.py)
- IPv6 disabled in WSL (/etc/sysctl.d/99-disable-ipv6.conf)

## Cooperation metric semantics (precise wording for writeup)
`multi_agent_reward_steps` counts timesteps on which at least two agents received
positive raw reward. Every successful multi-agent load produces such a timestep;
therefore a value of zero establishes that no successful multi-agent loads occurred.
A nonzero value is inconclusive: simultaneous independent pickups produce the same
signal, and multiple loading events can occur within one timestep. The bound holds
over timesteps, not events: cooperative-load timesteps <= multi_agent_reward_steps.
Zero does NOT mean "no coordination" - failed LOAD attempts and joint movement are
not detected. Exact per-load counter (food-removal diff + participant identification)
required before reporting any cooperation rate.
