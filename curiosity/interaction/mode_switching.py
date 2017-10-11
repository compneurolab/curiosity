'''
Utilities for various model and data provider transformations in switching training modes.

For example, say we initialize some models, and then change part of that initialization afterwards. Can do it here instead of worrying about involving dbinterface.
'''


def post_init_reinit_uncertainty_model(sess, updater):
    print('Reinitializing uncertainty model.')
    global_assign = updater.global_step.assign(0)
    sess.run([var.initializer for var in updater.um.var_list] + [global_assign])
    return updater






