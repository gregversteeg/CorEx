"""Correlation Explanation

A method to learn a hierarchy of successively more abstract
representations of complex data that are maximally
informative about the data. This method is unsupervised,
requires no assumptions about the data-generating model,
and scales linearly with the number of variables.

Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
High-Dimensional Data Through Correlation Explanation."
NIPS, 2014. arXiv preprint arXiv:1406.1222.

Code below written by Greg Ver Steeg and Gabriel Pereyra

License: GPL2
"""

import numpy as np  # Tested with 1.8.0
from scipy.misc import logsumexp  # Tested with 0.13.0

class Corex(object):
    """
    Correlation Explanation

    Going to try to follow sklearn naming/style (e.g. fit(X) to train)

    Attributes
    ----------

    n_hidden : int, default=2
        The number of hidden factors.

    dim_hidden : int, default=2
        The dimension of the hidden factors.

    p_y_given_x : array, [n_hidden, n_samples, dim_hidden]
        The distribution of latent factors for each sample.

    TODO: Add. Do I put every single output here as well?


    """
    def __init__(self, n_hidden=2, dim_hidden=2,            # Size of representations
                 batch_size=1e6, max_iter=400, n_repeat=1,  # Computational limits
                 eps=1e-6, alpha_hyper=(0.3, 1., 500.), balance=0.,     # Parameters
                 missing_values=-1, seed=None, verbose=False, more_verbose=False):

        self.dim_hidden = dim_hidden  # Each hidden factor can take dim_hidden discrete values
        self.n_hidden = n_hidden  # Number of hidden factors to use (Y_1,...Y_m) in paper
        self.missing_values = missing_values  # Implies the value for this variable for this sample is unknown

        self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence
        self.batch_size = batch_size  # TODO: re-implement running with mini-batches
        self.n_repeat = n_repeat  # TODO: Run multiple times and take solution with largest TC

        self.eps = eps  # Change in TC to signal convergence
        self.lam, self.tmin, self.ttc = alpha_hyper  # Hyper-parameters for updating alpha
        self.balance = balance # 0 implies no balance constraint. Values between 0 and 1 are valid.

        np.random.seed(seed)  # Set for deterministic results
        self.verbose = verbose
        self.more_verbose = more_verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print 'corex, rep size:', n_hidden, dim_hidden
        if more_verbose:
            np.seterr(all='warn')
        else:
            np.seterr(all='ignore')

    def label(self, p_y_given_x):
        """Maximum likelihood labels for some distribution over y's"""
        return np.argmax(p_y_given_x, axis=2).T

    @property
    def labels(self):
        """Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
        return self.label(self.p_y_given_x)

    @property
    def clusters(self):
        """Return cluster labels for variables"""
        return np.argmax(self.alpha[:,:,0],axis=0)

    @property
    def tc(self):
        """The total correlation explained by all the Y's.
        (Currently correct only for trees, modify for non-trees later.)"""
        return np.sum(self.tcs)

    def event_from_sample(self, x):
        """Transform data into event format.
        For each variable, for each possible value of dim_visible it could take (an event),
        we return a boolean matrix of True/False if this event occurred in this sample, x.
        Parameters:
        x: {array-like}, shape = [n_visible]
        Returns:
        x_event: {array-like}, shape = [n_visible * self.dim_visible]
        """
        x = np.asarray(x)
        n_visible = x.shape[0]
        assert self.n_visible == n_visible, \
            "Incorrect dimensionality for samples to transform."
        return np.ravel(x[:, np.newaxis] == np.tile(np.arange(self.dim_visible), (n_visible, 1)))

    def events_from_samples(self, X):
        """Transform data into event format. See event_from_sample docstring."""
        n_samples, n_visible = X.shape
        events_to_transform = np.empty((self.n_events, n_samples), dtype=bool)
        for l, x in enumerate(X):
            events_to_transform[:, l] = self.event_from_sample(x)
        return events_to_transform

    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        if X.ndim < 2:
            X = X[np.newaxis, :]
        events_to_transform = self.events_from_samples(X)
        p_y_given_x, log_z = self.calculate_latent(events_to_transform)
        if details:
            return p_y_given_x, log_z
        else:
            return self.label(p_y_given_x)

    def fit(self, X, **params):
        """Fit CorEx on the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_visible]
            Data matrix to be

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit corex on the data (this used to be ucorex)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.

        """

        self.initialize_parameters(X)

        X_event = self.events_from_samples(X)  # Work with transformed representation of data for efficiency

        self.p_x, self.entropy_x = self.data_statistics(X_event)
        
        for nloop in range(self.max_iter):

            self.update_marginals(X_event, self.p_y_given_x)  # Eq. 8

            if self.n_hidden > 1:  # Structure learning step
                self.mis = self.calculate_mis(self.log_p_y, self.log_marg)
                self.update_alpha(self.mis, self.tcs)  # Eq. 9

            self.p_y_given_x, log_z = self.calculate_latent(X_event)  # Eq. 7

            self.update_tc(log_z)  # Calculate TC and record history for convergence

            self.print_verbose()
            if self.convergence(): break

        self.sort_and_output(log_z)

        return self.labels

    def initialize_parameters(self, X):
        """Set up starting state

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_visible]
            The data.

        """

        self.n_samples, self.n_visible = X.shape
        self.initialize_events(X)
        self.initialize_representation()

    def initialize_events(self, X):
        values_in_data = set(np.unique(X).tolist())-set([self.missing_values])
        self.dim_visible = int(max(values_in_data)) + 1
        if not set(range(self.dim_visible)) == values_in_data:
            print "Warning: Data matrix values should be consecutive integers starting with 0,1,..."
        self.n_events = self.n_visible * self.dim_visible

    def initialize_representation(self):
        if self.n_hidden > 1:
            self.alpha = (0.5+0.5*np.random.random((self.n_hidden, self.n_visible, 1)))
        else:
            self.alpha = np.ones((self.n_hidden, self.n_visible, 1), dtype=float)
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)

        log_p_y_given_x_unnorm = -np.log(self.dim_hidden) * (0.5 + np.random.random((self.n_hidden, self.n_samples, self.dim_hidden)))
        #log_p_y_given_x_unnorm = -100.*np.random.randint(0,2,(self.n_hidden, self.n_samples, self.dim_hidden))
        self.p_y_given_x, log_z = self.normalize_latent(log_p_y_given_x_unnorm)

    def data_statistics(self, X_event):
        p_x = np.sum(X_event, axis=1).astype(float)
        p_x = p_x.reshape((self.n_visible, self.dim_visible))
        p_x /= np.sum(p_x, axis=1, keepdims=True)  # With missing values, each x_i may not appear n_samples times
        entropy_x = np.sum(np.where(p_x>0., -p_x * np.log(p_x), 0), axis=1)
        entropy_x = np.where(entropy_x > 0, entropy_x, 1e-10)
        return p_x, entropy_x

    def update_marginals(self, X_event, p_y_given_x):
        self.log_p_y = self.calculate_p_y(p_y_given_x)
        self.log_marg = self.calculate_p_y_xi(X_event, p_y_given_x) - self.log_p_y

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j) using a tiny bit of Laplace smoothing to avoid infinities."""
        pseudo_counts = 0.001 + np.sum(p_y_given_x, axis=1, keepdims=True)
        log_p_y = np.log(pseudo_counts) - np.log(np.sum(pseudo_counts, axis=2, keepdims=True))
        return log_p_y

    def calculate_p_y_xi(self, X_event, p_y_given_x):
        """Estimate log p(y_j|x_i) using a tiny bit of Laplace smoothing to avoid infinities."""
        pseudo_counts = 0.001 + np.dot(X_event, p_y_given_x).transpose((1,0,2))  # n_hidden, n_events, dim_hidden
        log_marg = np.log(pseudo_counts) - np.log(np.sum(pseudo_counts, axis=2, keepdims=True))
        return log_marg  # May be better to calc log p(x_i|y_j)/p(x_i), as we do in Marg_Corex

    def calculate_mis(self, log_p_y, log_marg):
        """Return normalized mutual information"""
        vec = np.exp(log_marg + log_p_y)  # p(y_j|x_i)
        smis = np.sum(vec * log_marg, axis=2)
        smis = smis.reshape((self.n_hidden, self.n_visible, self.dim_visible))
        mis = np.sum(smis * self.p_x, axis=2, keepdims=True)
        return mis/self.entropy_x.reshape((1, -1, 1))

    def update_alpha(self, mis, tcs):
        t = (self.tmin + self.ttc * np.abs(tcs)).reshape((self.n_hidden, 1, 1))
        maxmis = np.max(mis, axis=0)
        alphaopt = np.exp(t * (mis - maxmis))
        self.alpha = (1. - self.lam) * self.alpha + self.lam * alphaopt

    def calculate_latent(self, X_event):
        """"Calculate the probability distribution for hidden factors for each sample."""
        alpha_rep = np.repeat(self.alpha, self.dim_visible, axis=1)
        log_p_y_given_x_unnorm = (1. - self.balance) * self.log_p_y + np.transpose(np.dot(X_event.T, alpha_rep*self.log_marg), (1, 0, 2))

        return self.normalize_latent(log_p_y_given_x_unnorm)

    def normalize_latent(self, log_p_y_given_x_unnorm):
        """Normalize the latent variable distribution

        For each sample in the training set, we estimate a probability distribution
        over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
        This normalization factor is quite useful as described in upcoming work.

        Parameters
        ----------
        Unnormalized distribution of hidden factors for each training sample.

        Returns
        -------
        p_y_given_x : 3D array, shape (n_hidden, n_samples, dim_hidden)
            p(y_j|x^l), the probability distribution over all hidden factors,
            for data samples l = 1...n_samples
        log_z : 2D array, shape (n_hidden, n_samples)
            Point-wise estimate of total correlation explained by each Y_j for each sample,
            used to estimate overall total correlation.

        """

        log_z = logsumexp(log_p_y_given_x_unnorm, axis=2)  # Essential to maintain precision.
        log_z = log_z.reshape((self.n_hidden, -1, 1))

        return np.exp(log_p_y_given_x_unnorm - log_z), log_z

    def update_tc(self, log_z):
        self.tcs = np.mean(log_z, axis=1).reshape(-1)
        self.tc_history.append(np.sum(self.tcs))

    def sort_and_output(self, log_z):
        order = np.argsort(self.tcs)[::-1]  # Order components from strongest TC to weakest
        self.tcs = self.tcs[order]  # TC for each component
        self.alpha = self.alpha[order]  # Connections between X_i and Y_j
        self.p_y_given_x = self.p_y_given_x[order]  # Probabilistic labels for each sample
        self.log_marg = self.log_marg[order]  # Parameters defining the representation
        self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
        self.log_z = log_z[order]  # -log_z can be interpreted as "surprise" for each sample
        if hasattr(self, 'mis'):
            self.mis = self.mis[order]

    def print_verbose(self):
        if self.verbose:
            print self.tcs
        if self.more_verbose:
            print self.alpha[:,:,0]
            if hasattr(self, "mis"):
                print self.mis[:,:,0]

    def convergence(self):
        dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
        return np.abs(dist) < self.eps # Check for convergence. dist is nan for empty arrays, but that's OK

