
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import load_iris
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick

iris = sklearn.datasets.load_iris()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

exp = explainer.explain_instance(test[0], rf.predict_proba, num_features=4, top_labels=3)

figure = exp.as_html()
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig('export/single.pdf', format='pdf')


f = open('export/single.html','w')
f.write(figure)
f.close()

sp = submodular_pick.SubmodularPick(explainer, train, rf.predict_proba, sample_size=20, num_exps_desired=3)

i = 0
for exp in sp.sp_explanations:
    print('instance: ', exp.predict_proba)
    figure = exp.as_pyplot_figure()
    figure.tight_layout()
    figure.savefig('export/multi{}.pdf'.format(i), format='pdf')
    i += 1
