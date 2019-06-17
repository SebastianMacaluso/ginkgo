class Simulator:
    """ Abstract simulator interface. """

    def forward(self, inputs):
        raise NotImplementedError()

    def log_prob(self, inputs, outputs):
        raise IntractableException

    def __call__(self, inputs):
        return self.forward(inputs)



class IntractableException(Exception):
    pass
