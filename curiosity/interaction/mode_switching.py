'''
Utilities for various model and data provider transformations in switching training modes.

For example, say we initialize some models, and then change part of that initialization afterwards. Can do it here instead of worrying about involving dbinterface.
'''
import tensorflow as tf

def post_init_reinit_uncertainty_model(sess, updater):
    print('Reinitializing uncertainty model.')
    #global_assign = updater.global_step.assign(0)
    sess.run([var.initializer for var in updater.um.var_list])
    return updater

def panic_reinit(sess, updater):
    print('ALL GLOBALS')
    print([var.name for var in tf.global_variables()])
    adam_var_init = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    global_assign = tf.assign(updater.global_step, 0)
    sess.run(adam_var_init + [global_assign])
    return updater





