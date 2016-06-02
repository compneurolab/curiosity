import os

def main(ind, srcdir, savedir, gpu=0):
    learningrate = 1.
    for i in range(10):
        seed = ind
        experiment_id = "seed:%d_learningrate:%f" % (seed, learningrate)
        cmd_tmp = """CUDA_VISIBLE_DEVICES=%d python %s/curiosity/curiosity/sandbox/normal_encoder_opt_source.py %s --seed %d --learningrate=%f --savedir=%s"""
        cmd = cmd_tmp % (gpu, srcdir, experiment_id, seed, learningrate, savedir)
        os.system(cmd)
        learningrate = learningrate / 2.
