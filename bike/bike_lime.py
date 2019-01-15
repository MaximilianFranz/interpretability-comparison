
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick
from load_bike_data import load_bike

X, Y, names = load_bike()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)

rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500)
rf.fit(train, labels_train)

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=names, class_names=['bike_count'], discretize_continuous=True, mode='regression')

exp = explainer.explain_instance(test[0], rf.predict, num_features=4, top_labels=3)

figure = exp.as_html()
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig('../export/bike/single.pdf', format='pdf')


f = open('../export/bike/single.html','w')
f.write(figure)
f.close()

# sp = submodular_pick.SubmodularPick(explainer, train, rf.predict_proba, sample_size=20, num_exps_desired=3)
#
# i = 0
# for exp in sp.sp_explanations:
#     print('instance: ', exp.predict_proba)
#     figure = exp.as_pyplot_figure()
#     figure.tight_layout()
#     figure.savefig('export/multi{}.pdf'.format(i), format='pdf')
#     i += 1
