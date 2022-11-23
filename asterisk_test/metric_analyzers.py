"""
Several analyzer classes which combine metric data for different sets of data and exports them. [NOT DONE]
"""



import pandas as pd
from scipy import stats
from pathlib import Path

from ast_hand_translation import AstHandTranslation
from ast_hand_rotation import AstHandRotation
from file_manager import my_ast_files


class AstDirAnalyzer:
    """
    This class takes trials in one direction and stores (and saves) the metrics together
    """
    def __init__(self, trials, file_obj, avg=None):
        self.file_locs = file_obj
        self.t_dir = trials[0].trial_translation
        self.r_dir = trials[0].trial_rotation
        self.hand_name = trials[0].hand.get_name()

        metric_df = pd.DataFrame()
        for t in trials:  # TODO: what about trials  with no_mvt labels, metrics aren't calculated on them
            if "no_mvt" in t.path_labels:
                continue

            metric_df = metric_df.append(t.metrics, ignore_index=True)

        if len(metric_df) > 0:
            metric_df = metric_df.set_index("trial")
            #print(metric_df)
            self.metrics = metric_df
        else:
            self.metrics = None

        # print(f"{self.t_dir}_{self.r_dir}")
        # save trial objects in case we need it
        self.avg = avg
        self.trials = trials

    def save_data(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """
        if file_name_overwrite is None:
            new_file_name = self.file_locs.metric_results / f"{self.hand_name}_{self.t_dir}_{self.r_dir}_results.csv"
        else:
            new_file_name = self.file_locs.metric_results / file_name_overwrite + ".csv"

        if self.metrics is not None:
            self.metrics.to_csv(new_file_name, index=True)


class AstHandAnalyzer:
    """
    Takes a hand data object and gets all the metrics
    """
    def __init__(self, file_obj, hd, do_avg_line_metrics=True):
        self.hand_name = hd.hand.get_name()
        self.hand_data = hd  # keeping it around just in case

        self.file_locs = file_obj

        # make a dir analyzer for each direction
        dir_analyzers = []
        complete_df = pd.DataFrame()

        for key in hd.data.keys():
            trials = hd.data[key]
            # TODO: implement average later
            analyzer = AstDirAnalyzer(trials, self.file_locs)  # TODO: can we check for no metrics? or for no_mvt?
            complete_df = complete_df.append(analyzer.metrics)
            if analyzer.metrics is not None:
                dir_analyzers.append(analyzer)

        self.analyzers = dir_analyzers
        # print(complete_df)
        # complete_df = complete_df.set_index("trial")
        self.all_metrics = complete_df

        avg_df = pd.DataFrame()
        avg_sd_df = pd.DataFrame()
        all_avg_metrics = pd.DataFrame()
        for avg in hd.averages:
            avg_df = avg_df.append(avg.metrics_avgd, ignore_index=True)  # should have values in these
            avg_sd_df = avg_sd_df.append(avg.metrics_avgd_sds, ignore_index=True)

            # TODO: its stupid, but its a workaround for rotation avgs right now... see if we can make mor elegant
            if do_avg_line_metrics or "no_mvt" in avg.path_labels:
                all_avg_metrics = all_avg_metrics.append(avg.metrics, ignore_index=True)

        avg_df = avg_df.set_index("trial")
        avg_sd_df = avg_sd_df.set_index("trial")

        if do_avg_line_metrics or len(all_avg_metrics) > 0:
            all_avg_metrics = all_avg_metrics.set_index("trial")

        self.metrics_avgd = avg_df
        self.metrics_avgd_sds = avg_sd_df

        if do_avg_line_metrics or len(all_avg_metrics) > 0:
            self.all_avg_metrics = all_avg_metrics
        else:
            self.all_avg_metrics = None

    def save_data(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """

        if self.all_avg_metrics is not None:
            names = ["trial_metrics", "avg_trial_metrics", "trial_metrics_avgd", "trial_metric_avgd_sds"]
            data = [self.all_metrics, self.all_avg_metrics, self.metrics_avgd, self.metrics_avgd_sds]
        else:
            names = ["trial_metrics", "trial_metrics_avgd", "trial_metric_avgd_sds"]
            data = [self.all_metrics, self.metrics_avgd, self.metrics_avgd_sds]

        for n, d in zip(names, data):
            if file_name_overwrite is None:
                new_file_name = self.file_locs.metric_results / f"{self.hand_name}_{n}.csv"
            else:
                new_file_name = self.file_locs.metric_results / f"{file_name_overwrite}_{n}.csv"

            d.to_csv(new_file_name, index=True)


if __name__ == '__main__':
    h = "2v2"
    rot = "n"
    subjects = ["sub1", "sub2", "sub3"]
    translation = False

    print(f"Getting {h} ({rot}) data...")
    if translation:
        data = AstHandTranslation(subjects, h, rotation=rot, blocklist_file="trial_blocklist.csv")
        data.filter_data(10)
        data.calc_averages()
        results = AstHandAnalyzer(data, my_ast_files)
    else:
        data = AstHandRotation(subjects, h)
        data.filter_data(10)
        data.calc_averages()
        results = AstHandAnalyzer(data, my_ast_files, do_avg_line_metrics=False)

    print(f"Average Metrics: {results.metrics_avgd}")
    print(f"Standard deviations of average metrics: {results.metrics_avgd_sds}")
    print(f"Metrics of the average lines: {results.all_avg_metrics}")
