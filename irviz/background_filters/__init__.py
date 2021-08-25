from .emsc import *
from .gpr import *

# Standards for contribution
# - Filters should avoid raising errors; errors will be printed to the user on every execution
# - Filters should be added to testing (see irviz.tests.test_filters)
# - Code should be well formatted, either with Pycharm's formatter (Ctrl+Alt+L?) or Black; line-lengths < 120 ideally
# - Code should be cleaned with linter (see yellow squiggle underlines in pycharm)
# - Doc strings required for public API components
# - Contextual comments throughout code, particularly for high-complexity lines
# - Avoid unnecessary use of external packages where possible to limit dependency bloat
# - Utilize dask laziness when possible/appropriate to minimize memory footprint in support of large datasets
