from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy as NoSahreCnnPolicy
from baselines.ppo1.cnn_policy import CnnPolicy

def policy_network(policy_type):

    if policy_type == 'mlp':
        policy_fn = lambda name, ob_space, ac_space: MlpPolicy(name=name, 
                ob_space=ob_space, ac_space=ac_space,
                hid_size=32, num_hid_layers=2)
    elif policy_type == 'cnn':
        policy_fn = lambda name, ob_space, ac_space: CnnPolicy(name=name,
                ob_space=ob_space, ac_space=ac_space)
    elif policy_type =='noshare_cnn':
        policy_fn = lambda name, ob_space, ac_space: NoSahreCnnPolicy(name=name,
                ob_space=ob_space, ac_space=ac_space)
    else:
        raise NotImplementedError 
    return policy_fn
