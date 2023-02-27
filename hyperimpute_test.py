from hyperimpute.plugins.imputers import Imputers
import numpy as np
import pandas as pd

imputers = Imputers()

imputers.list()
method = 'hyperimpute'
plugin = Imputers().get(method)

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, 4], [3, 3, 9, 9], [2, 2, 2, 2]])

print(np.sum(X.isna()))
out = plugin.fit_transform(X)