from results.data.store import load
from correlations.errors import uWerr

dest = 'results/data/numpy_objs/errs_xx_kg_cfn.json'
a = load(dest)

f_aav, f_diff, f_ddiff, itau, itau_diff, itau_aav = uWerr(a)
print f_aav, f_diff, f_ddiff, itau, itau_diff