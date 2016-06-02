import os

def main(ind, gpu=0):
    learningrate = 1.
    for i in range(10):
        seed = ind
        experiment_id = "seed:%d_learningrate:%f" % (seed, learningrate)
        cmd_tmp = """CUDA_VISIBLE_DEVICES=%d python /om/user/yamins/src/curiosity/curiosity/sandbox/normal_encoder_opt_source.py %s --seed %d --learningrate=%f --savedir=/om/user/yamins/tensorflow_checkpoint_cache"""
        cmd = cmd_tmp % (gpu, experiment_id, seed, learningrate)
        os.system(cmd)
        learningrate = learningrate / 2.
