from InferenceHelpers import functions


class FitInspector(object):
    """Docstring for FitInspector. """

    def __init__(self, **kwargs):
        """TODO: to be defined. """
        self.df_fixed, self.df_uncond = functions.check_fitting_details(**kwargs)
        self.LLR = self.df_fixed["fval"] - self.df_uncond["fval"]
        self.cols = functions.derive_columns(self.df_uncond)

    def plot_2D_distributions(self, col=None):
        functions.plot_2D_distributions(df_uncond=self.df_uncond, df_fixed=self.df_fixed, LLR=self.LLR, col=col, cols=self.cols)

    def plot_1D_distributions(self, col=None):
        functions.plot_1D_distributions(df_uncond=self.df_uncond, df_fixed=self.df_fixed, cols=self.cols, col=col)
