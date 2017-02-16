# Tests the USPS handwritten database


import numpy as np
import test_rbfopt_env
import rbfopt_settings
import rbfopt_algorithm
import usps_blackbox
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pickle


def rbfopt_search(rand_seed=np.random.randint(2 ** 30)):

    black_box = usps_blackbox.Blackbox()
    settings = rbfopt_settings.RbfSettings(num_cpus=1,
                                           max_stalled_cycles=100,
                                           max_iterations=500,
                                           max_evaluations=1000,
                                           global_search_method='genetic',
                                           init_global_search_method='genetic',
                                           init_strategy='quasilhd',
                                           domain_scaling='off',
                                           max_consecutive_local_searches=1,
                                           num_global_searches=5,
                                           print_solver_output=True,
                                           rand_seed=rand_seed)

    alg = rbfopt_algorithm.OptAlgorithm(settings=settings, black_box=black_box)

    x_min, point, itercount, evalcount, fast_evalcount = alg.optimize()

    #print x_min
    #print point
    #print sum(point)

    return x_min


def predictor():

    usps = pickle.load(open('usps.pickle', 'rb'))

    X = usps['X']
    Y = np.ravel(usps['Y'])

    X_train = X[:7291, :]
    Y_train = Y[:7291]
    X_validate = X[7291:, :]
    Y_validate = Y[7291:]

    # Decision Tree
    clf = tree.DecisionTreeClassifier()

    # Neural Network
    #clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,))

    tot_err = []

    for j in range(10):

        err = []
        print 'Fitting the predictor...'
        for i in range(500):
            r_idx = np.random.choice(256, 15, replace=False)
            r = np.zeros(256, dtype=bool)
            r[r_idx] = 1
            # r = np.random.randint(0, 2, 256).astype(bool)
            clf.fit(X_train[:, r], Y_train)
            prediction = clf.predict(X_validate[:, r])

            err.append(sum(prediction != Y_validate))

        tot_err.append(err)
        print(min(err))
    pickle.dump(tot_err, open('err_dt_random.pickle', 'wb'))


def feature_selection():

    usps = pickle.load(open('usps.pickle', 'rb'))

    X = usps['X']
    Y = np.ravel(usps['Y'])

    X_train = X[:7291, :]
    Y_train = Y[:7291]
    X_validate = X[7291:, :]
    Y_validate = Y[7291:]

    #print X_train.shape[1]

    clf = ExtraTreesClassifier()
    clf.fit(X_train, Y_train)
    print clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_train)
    print X_new.shape[1]
    r = model.get_support(indices=True)
    print r

    fi = clf.feature_importances_
    print(fi)
    sfi = np.argsort(fi)
    print(sfi)
    r = sfi[-15:]
    print(np.sort(r))

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train[:, r], Y_train)
    prediction = clf.predict(X_validate[:, r])

    err = (1 - sum(prediction != Y_validate) / 2007.0) * 100
    print(err)


if (__name__ == '__main__'):
    rbfopt_search(1)
    #predictor()
    #feature_selection()