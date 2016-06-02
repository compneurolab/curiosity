import os

def main(ind, srcdir, savedir, gpu=0, script='normal_encoder_opt_source.py'):
    learningrate = 1.
    for i in range(10):
        seed = ind
        experiment_id = "seed:%d_learningrate:%f" % (seed, learningrate)
        cmd_tmp = """CUDA_VISIBLE_DEVICES=%d python %s/curiosity/curiosity/sandbox/%s %s --seed %d --learningrate=%f --savedir=%s"""
        cmd = cmd_tmp % (gpu, srcdir, script, experiment_id, seed, learningrate, savedir)
        os.system(cmd)
        learningrate = learningrate / 2.
