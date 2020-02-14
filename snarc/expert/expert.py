class Expert(object):
    """
    A solver for a very specific class of situations.

    You train it on a handful of input-output examples, then use it to solve
    similar problems.
    """

    def fit(self, xx, yy):
        """
        Attempt to learn input -> output transform.  Returns true iff success.
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Transform the input as learned by fit().
        """
        raise NotImplementedError
