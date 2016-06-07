import os
import pymongo as pm
conn = pm.MongoClient(port=29101)

num_tries = 10
def main(ind, dbname, colname, srcdir, savedir, gpu=0, script='normal_encoder_opt_source.py', decayrate=0.95, decaystep=100000):
    for n in range(num_tries):
        learningrate = 1.
        print('reset learning rate to %f' % learningrate)
        for i in range(10):
            seed = ind * num_tries + n
            print('Seed: %d' % seed)
            experiment_id = "seed:%d_learningrate:%f_decaystep:%d_decayrate:%f" % (seed, learningrate, decaystep, decayrate)
            if conn[dbname][colname].find({'experiment_id': experiment_id}).count() > 10:
                print('Breaking out at %s' % experiment_id)
                break
            cmd_tmp = """CUDA_VISIBLE_DEVICES=%d python %s/curiosity/curiosity/sandbox/%s %s %s %s --seed %d --learningrate=%f --savedir=%s --decaystep=%d --decayrate=%f"""
            cmd = cmd_tmp % (gpu, srcdir, script, dbname, colname, experiment_id, seed, learningrate, savedir, decaystep, decayrate)
            os.system(cmd)
            if conn[dbname][colname].find({'experiment_id': experiment_id}).count() > 10:
                print('Breaking out at %s due to enough steps' % experiment_id)
                break
            elif conn[dbname][colname].find({'experiment_id': experiment_id}).distinct('step') == [-1]:
                print('Breaking out at %s due to no steps' % experiment_id)
                break
            learningrate = learningrate / 2.
