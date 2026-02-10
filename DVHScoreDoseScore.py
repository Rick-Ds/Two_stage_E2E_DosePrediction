"""
DVHScoreDoseScore.py

OpenKBP-style dose evaluation utilities.

EvaluateDose computes per-ROI DVH endpoints (targets: D_99/D_95/D_1; OARs: D_0.1cc/mean) and derives:
- DVH score: mean absolute difference across all DVH endpoints
- Dose score: mean absolute voxel-wise difference within the possible-dose region

Author: Boda Ning
"""

from itertools import product as it_product

import numpy as np
import pandas as pd
import tqdm
from torch.utils import data


class EvaluateDose:

    def __init__(self, data_src):

        self.data_src = data_src


        self.patient_id = None
        self.roi_mask = None
        self.new_dose = None

        self.voxel_size = None
        self.possible_dose_mask = None
        self.dose_score_vec = np.zeros(len(self.data_src))
        self.sample_idx = 0


        self.oar_eval_metrics = ['D_0.1_cc', 'mean']
        self.tar_eval_metrics = ['D_99', 'D_95', 'D_1']


        oar_metrics = list(it_product(self.oar_eval_metrics, self.data_src.rois['oars']))
        target_metrics = list(it_product(self.tar_eval_metrics, self.data_src.rois['targets']))


        self.metric_difference_df = pd.DataFrame(index=self.data_src.patient_id_list,
                                                 columns=[*oar_metrics, *target_metrics])
        self.dose_a_metric_df = self.metric_difference_df.copy()
        self.dose_b_metric_df = self.metric_difference_df.copy()

    def append_sample(self, dose_arr_a, batch, dose_arr_b=None):


        batch_size = dose_arr_a.shape[0]

        for i in range(batch_size):


            self.roi_mask = batch['structure_masks'][i].numpy().astype(bool)


            self.patient_id = batch['patient_id'][i]


            self.voxel_size = np.prod(batch['voxel_dimensions'][i].numpy())


            self.possible_dose_mask = batch['possible_dose_mask'][i].numpy()

            dose_a = dose_arr_a[i].flatten()
            self.dose_a_metric_df = self.calculate_metrics(self.dose_a_metric_df, dose_a)

            if dose_arr_b is not None:
                dose_b = dose_arr_b[i].flatten()
                self.dose_b_metric_df = self.calculate_metrics(self.dose_b_metric_df, dose_b)
                self.dose_score_vec[self.sample_idx] = np.sum(np.abs(dose_a - dose_b)) / np.sum(self.possible_dose_mask)
                self.sample_idx += 1

    def make_metrics(self):
        dvh_score = np.nanmean((np.abs(self.dose_a_metric_df - self.dose_b_metric_df).values))
        dose_score = self.dose_score_vec.mean()
        return dvh_score, dose_score

    def get_patient_dose_tensor(self, dose_batch):


        dose_key = [key for key in dose_batch.keys() if 'dose' in key.lower()][0]
        dose_tensor = dose_batch[dose_key][0].numpy()
        dose_tensor = dose_tensor.flatten()
        return dose_tensor

    def get_constant_patient_features(self, rois_batch):


        self.roi_mask = rois_batch['structure_masks'][0].numpy().astype(bool)

        self.patient_list = rois_batch['patient_id']

        self.voxel_size = np.prod(rois_batch['voxel_dimensions'].numpy())

        self.possible_dose_mask = rois_batch['possible_dose_mask'].numpy()

    def calculate_metrics(self, metric_df, dose):

        roi_exists = self.roi_mask.max(axis=(1, 2, 3))

        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / self.voxel_size))
        for roi_idx, roi in enumerate(self.data_src.full_roi_list):
            if roi_exists[roi_idx]:


                roi_mask = self.roi_mask[roi_idx, :, :, :].flatten()
                roi_dose = dose[roi_mask]
                roi_size = len(roi_dose)

                if roi in self.data_src.rois['oars']:
                    if 'D_0.1_cc' in self.oar_eval_metrics:

                        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / roi_size * 100
                        metric_eval = np.percentile(roi_dose, fractional_volume_to_evaluate)
                        metric_df.at[self.patient_id, ('D_0.1_cc', roi)] = metric_eval
                    if 'mean' in self.oar_eval_metrics:
                        metric_eval = roi_dose.mean()
                        metric_df.at[self.patient_id, ('mean', roi)] = metric_eval
                elif roi in self.data_src.rois['targets']:
                    if 'D_99' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 1)
                        metric_df.at[self.patient_id, ('D_99', roi)] = metric_eval
                    if 'D_95' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 5)
                        metric_df.at[self.patient_id, ('D_95', roi)] = metric_eval
                    if 'D_1' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 99)
                        metric_df.at[self.patient_id, ('D_1', roi)] = metric_eval

        return metric_df
