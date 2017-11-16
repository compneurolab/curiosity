"""
Data configs
"""
from config_base import DataConfig

class SequenceDataProvider(DataConfig):
    def __init__():


"""
Eight World Data: Recorded data from ThreeDWorld to train Flex Models
"""
eight_world_data = DataConfig(
	train_path = '/mnt/fs1/datasets/eight_world_dataset/new_tfdata',
        num_train_examples = 84 * 256 * 4,
        validation_path = '/mnt/fs1/datasets/eight_world_dataset/new_tfvaldata',
        num_validation_examples = 12 * 256 * 4,
        )
eight_world_data.stats_file = '/mnt/fs1/datasets/eight_world_dataset/' + \
        'new_stats/stats_std.pkl'

"""
Eight World Data Local: Local copy of Eight World Data
"""
eight_world_data_local = DataConfig(
	train_path = '/data2/mrowca/datasets/eight_world_dataset/new_tfdata',
        num_train_examples = 84 * 256 * 4,
        validation_path = '/data2/mrowca/datasets/eight_world_dataset/new_tfvaldata',
        num_validation_examples = 12 * 256 * 4,
        )
eight_world_data_local.stats_file = '/data2/mrowca/datasets/eight_world_dataset/' + \
        'new_stats/stats_std.pkl'
