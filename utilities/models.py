import numpy as np
from functools import partial
from spiketools.stats.anova import create_dataframe, create_dataframe_bins, fit_anova
from spiketools.spatial.information import compute_spatial_information
###################################################################################################
###################################################################################################
PLACE_MODELS = {
    'MODEL' : 'fr ~ C(bin)',
    'FEATURE' : 'C(bin)',
    'COLUMNS' : ['bin', 'fr']
}

create_df_place = partial(create_dataframe_bins,bin_columns=PLACE_MODELS['COLUMNS'])
fit_anova_place = partial(fit_anova, formula=PLACE_MODELS['MODEL'], feature=PLACE_MODELS['FEATURE'])


# PLACE_MODELS = {
#     'MODEL' : 'fr ~ C(bin)',
#     'FEATURE' : 'C(bin)',
#     'COLUMNS' : ['bin', 'fr']
# }

# create_df_place = partial(create_dataframe_bins, bin_columns=PLACE_MODELS['COLUMNS'])
# fit_anova_place = partial(fit_anova, formula=PLACE_MODELS['MODEL'],
#                           feature=PLACE_MODELS['FEATURE'])