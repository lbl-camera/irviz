from dash import Input, Output

from ryujin.utils import targeted_callback
from .spectra_plot import SpectraPlotGraph


class RegionSpectraPlot(SpectraPlotGraph):
    def init_callbacks(self, app):
        super(RegionSpectraPlot, self).init_callbacks(app)

    def _update_regions(self, data):
        # do nothing special if initing
        if hasattr(self, 'figure'):

            # clear shapes except _region_start
            self.figure.layout.shapes = []

            # repopulate from regionlist
            for region_record in data:
                region_min, region_max = region_record.get('region_min'), region_record.get('region_max')
                if region_min is not None and region_max is not None:
                    self.figure.add_vrect(region_min, region_max, line_width=0, opacity=.3, fillcolor='gray')
                elif region_max is None:
                    self.figure.add_vline(region_min, name='_region_start', line_color="gray")

        return self._update_figure()

