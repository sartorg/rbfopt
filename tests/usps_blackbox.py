"""Black-box USPS function.

"""
try:
    import cython_rbfopt.rbfopt_utils as ru
except ImportError:
    import rbfopt_utils as ru
import rbfopt_black_box as bb
import numpy as np
import pickle
from sklearn import tree
from sklearn.neural_network import MLPClassifier


class Blackbox(bb.BlackBox):
    """

    Attributes
    ----------

    dimension : int
        Dimension of the problem.

    var_lower : 1D numpy.ndarray[float]
        Lower bounds of the decision variables.

    var_upper : 1D numpy.ndarray[float]
        Upper bounds of the decision variables.

    integer_vars : 1D numpy.ndarray[int]
        A list of indices of the variables that must assume integer
        values.

    See also
    --------
    :class:`rbfopt_black_box.BlackBox`
    """

    def __init__(self):
        """Constructor.
        """

        usps = pickle.load(open('usps.pickle', 'rb'))

        X = usps['X']
        Y = np.ravel(usps['Y'])

        self.X_train = X[:7291, :]
        self.Y_train = Y[:7291]
        self.X_validate = X[7291:, :]
        self.Y_validate = Y[7291:]

        # Set required data
        self.dimension = 256

        self.var_lower = np.zeros(self.dimension)
        self.var_upper = np.ones(self.dimension)

        self.integer_vars = np.arange(256)

        self.A = np.array([[1]*256, [-1]*256])
        self.b = np.array([15, -5])
    # -- end function

    def get_dimension(self):
        """Return the dimension of the problem.

        Returns
        -------
        int
            The dimension of the problem.
        """
        return self.dimension
    # -- end function

    def get_var_lower(self):
        """Return the array of lower bounds on the variables.

        Returns
        -------
        1D numpy.ndarray[float]
            Lower bounds of the decision variables.
        """
        return self.var_lower
    # -- end function

    def get_var_upper(self):
        """Return the array of upper bounds on the variables.

        Returns
        -------
        1D numpy.ndarray[float]
            Upper bounds of the decision variables.
        """
        return self.var_upper
    # -- end function

    def get_integer_vars(self):
        """Return the list of integer variables.

        Returns
        -------
        1D numpy.ndarray[int]
            A list of indices of the variables that must assume
            integer values. Can be empty.
        """
        return self.integer_vars
    # -- end function

    def get_constraints(self):
        """Return the list of integer variables.

        Returns
        -------
        2D numpy.ndarray[int]
            The constraint matrix A in the system Ax <= b.
        """
        return self.A

    # -- end function

    def get_rhs(self):
        """Return the list of integer variables.

        Returns
        -------
        1D numpy.ndarray[int]
            The rhs b in the system Ax <= b.
        """
        return self.b

    # -- end function

    def evaluate(self, x):

        """Evaluate the black-box function.

        Parameters
        ----------
        x : 1D numpy.ndarray[float]
            Value of the decision variables.

        Returns
        -------
        float
            Value of the function at x.

        """
        assert (isinstance(x, np.ndarray))
        # Decision Tree
        clf = tree.DecisionTreeClassifier()

        # Neural Network
        # clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,))

        clf.fit(self.X_train[:, x.astype(bool)], self.Y_train)
        prediction = clf.predict(self.X_validate[:, x.astype(bool)])

        return sum(prediction != self.Y_validate) + \
            max(sum(x)-self.b[0], 0)**2 + max(-self.b[1]-sum(x), 0)**2
    # -- end function

    def evaluate_fast(self, x):
        """Evaluate a fast approximation of the black-box function.

        Returns an approximation of the value of evaluate(), hopefully
        much more quickly. If has_evaluate_fast() returns False, this
        function will never be queried and therefore it does not have
        to return any value.

        Parameters
        ----------
        x : 1D numpy.ndarray[float]
            Value of the decision variables.

        Returns
        -------
        float
            Approximate value of the function at x.

        """
        raise NotImplementedError('evaluate_fast not available')
    # -- end function

    def has_evaluate_fast(self):
        """Indicate whether evaluate_fast is available.

        Indicate if a fast but potentially noisy version of evaluate
        is available through the function evaluate_fast. If True, such
        function will be used to try to accelerate convergence of the
        optimization algorithm. If False, the function evaluate_fast
        will never be queried.

        Returns
        -------
        bool
            Is evaluate_fast available?
        """
        return False
    # -- end function

# -- end class
