"""
Set of base configurations for tfutils
"""
import collections

class DataConfig(object):
    """
    Data provider configurations
    """
    def __init__(self, 
            train_path, 
            num_train_examples, 
            validation_path, 
            num_validation_examples,
            sources=[],
            ):

        self.train_path = train_path
        self.num_train_examples = num_train_examples
        self.validation_path = validation_path
        self.num_validation_examples = num_validation_examples

        if sources:
            self.sources = sources
        else:
            self.sources = [source for source in os.listdir(self.train_path)
                    if os.path.isdir(os.path.join(self.train_path, source))]
            val_sources = [source for source in os.listdir(self.validation_path)
                    if os.path.isdir(os.path.join(
                        self.validation_path, source))]
            assert collections.Counter(self.sources) == \
                    collections.Counter(val_sources),
                    'Train and validation sources are not the same! %s vs %s' %
                    (self.sources, val_sources)
