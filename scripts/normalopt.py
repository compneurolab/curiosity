import os

num_tries = 10
def main(ind, dbname, colname, srcdir, savedir, gpu=0, script='normal_encoder_opt_source.py', decayrate=0.95, decaystep=100000):
    learningrate = 1.
    for n in range(num_tries):
        for i in range(10):
            seed = ind * num_tries + n
            experiment_id = "seed:%d_learningrate:%f_decaystep:%d_decayrate:%f" % (seed, learningrate, decaystep, decayrate)
            cmd_tmp = """CUDA_VISIBLE_DEVICES=%d python %s/curiosity/curiosity/sandbox/%s %s %s %s --seed %d --learningrate=%f --savedir=%s --decaystep=%d --decayrate=%f"""
            cmd = cmd_tmp % (gpu, srcdir, script, dbname, colname, experiment_id, seed, learningrate, savedir, decaystep, decayrate)
            os.system(cmd)
            learningrate = learningrate / 2.
