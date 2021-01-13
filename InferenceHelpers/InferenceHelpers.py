import os
from InferenceHelpers import functions

class InferenceHelpers(object):

    """Docstring for MyClass. """

    def __init__(self, pattern):
        self.pattern = pattern  # this is the pattern for the likelihood scan
        self.local_scan = None
        self.local_threshold = None

    def get_data_from_dali(self, pattern, fetch=False, threshold=False, dali_user="twolf", target="/Users/twolf/Physics/Xenon/NT_projections/MoMa"):
        if threshold:
            splitted_pattern = pattern.split("/")
            splitted_pattern[-2] = splitted_pattern[-2] + "_threshold"
            this_pattern = "/".join(splitted_pattern)
        else:
            this_pattern = pattern

        local_output = this_pattern.split("/")[-2]
        local_output = os.path.join(target, local_output)
        if not os.path.exists(local_output):
            os.makedirs(local_output)
        cmd = f"scp {dali_user}@dali2:{this_pattern} {local_output}"
        print(cmd)
        if fetch:
            os.system(cmd)

        print("data now in -->", local_output)
        return local_output

    def get_all_data_from_dali(self, fetch=False, dali_user="twolf", target="/Users/twolf/Physics/Xenon/NT_projections/MoMa"):
        self.local_scan = self.get_data_from_dali(pattern=self.pattern, fetch=fetch, threshold=False, dali_user=dali_user, target=target)
        self.local_threshold = self.get_data_from_dali(pattern=self.pattern, fetch=fetch, threshold=True, dali_user=dali_user, target=target)
        if fetch:
            print("data fetched!")
        return self.local_scan, self.local_threshold

    def plot_likelihood_scan(self, **kwargs):
        self.df_scan = functions.plot_likelihood_scan(local_path=self.local_scan, **kwargs)

    def plot_threshold(self, **kwargs):
        self.df_thres = functions.plot_threshold(local_path=self.local_threshold, **kwargs)

    def show_available_scans(self):
        if self.local_scan is not None:
            for item in os.listdir(self.local_scan):
                print(item)
