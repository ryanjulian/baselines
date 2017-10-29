# GAIL

- Original paper: https://arxiv.org/abs/1606.03476

## If you want to train an imitation learning agent

### Step 1: Generate expert data

#### train the trpo agent

Train the agent wiith trpo and save the final policy.
```bash
cd $BASELINES/baselines/trpo_mpi
# run the trpo for xxx timesteps
python run_mujoco.py --env_id $ENV_ID
```

### sample data from the learned agent

Sample the trajectories from the final policy.
```bash
cd $BASELINES/baselines/trpo_mpi
python run_mujoco.py --env_id $ENV_ID --load_model_path $CKPT_PATH
```

### Step 2: Imitation learning

You can move the pikcle from generate in Step 1 to `$BASELINES/baselines/gail/data` for convenience.

```bash
cd $BASELINES/baselines/gail
python run_mujoco.py --env_id $ENV_ID --expert_path $PICKLE_PATH 
```

See help (`-h`) for more options.

