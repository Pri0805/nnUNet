from nnunetv2.training.loss.compound_losses import CE_and_Tversky_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from nnunetv2.training.loss.robust_ce_loss import TverskyLoss


class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    def _build_loss(self):
        tversky_kwargs = {
            'alpha': 0.3,
            'beta': 0.7,
            'smooth': 1e-6,
            'ignore_index': self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        }

        loss = TverskyLoss(**tversky_kwargs)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # We give each output a weight which decreases exponentially (division by 2) as the resolution decreases.
            # This gives higher resolution outputs more weight in the loss.
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # We don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()

            # Now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerTverskyCELoss(nnUNetTrainer):
    def _build_loss(self):
        tversky_kwargs = {'alpha': 0.3, 'beta': 0.7, 'smooth': 1e-6}
        ce_kwargs = {}
        loss = CE_and_Tversky_loss(
            tversky_kwargs=tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1,
            weight_tversky=1,
            ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
