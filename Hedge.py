# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Hedge Defence`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)


class HedgeDefencePyTorch(Preprocessor):
    """
        Implementation of Hedge Defence as presented in the article : https://arxiv.org/pdf/2106.04938.pdf
        The implementation is for PyTorch models.

    """

    params = [
        "estimator",
        "loss",
        "num_classes",
        "step_size",
        "epsilon",
        'num_iter'
    ]

    def __init__(
            self,
            estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
            num_classes: int = 10,
            step_size: float = 0.01,
            epsilon: float = 0.03,
            num_iter: int = 20,
            apply_fit: bool = False,
            apply_predict: bool = True,
    ):
        """
        Initialize a Hedge Defence object.

        :param estimator: A trained classifier (PyTotrch).
        :param num_classes: The number of possible classes in the dataset.
        :param step_size: the step size in each iteration of the gradient ascending algorith.
        :param num_iter: The number of iterations the sample will be updated in the gradient ascending process.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.estimator = estimator
        self.loss = estimator.loss
        self.num_classes = num_classes
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_iter = num_iter
        self._check_params()

    def compute_all_classes_lost(self, output):
        """
        Compute the sum of losses to all classes. Basically it means attacking all classes and compute the loss for each one.
        :param output: the output of the model after processing the input sample.
        :return: A tesnor holding the sum of losses.
        """
        self.estimator.model.zero_grad()
        sum_losses = 0
        labels = [0] * self.num_classes  # each time we compute the loss against a different class.
        for i in range(self.num_classes):
            labels[i] = 1
            labels_tensor = torch.Tensor(labels).unsqueeze(0).to(self.estimator._device)
            cost = self.loss(output, labels_tensor)
            sum_losses += cost
            labels[i] = 0
        return torch.Tensor(sum_losses)

    def gradient_ascending(self, input_sample):
        """
        Gradient ascending algorithm - Iteratively update the input sample according to the gradien in order to get the perturbation that maximizes
        the sum of losses.
        :param input_sample: the input sample the model gets.
        :return: the updated input sample (numpy array).
        """
        input_sample.requires_grad_(True)

        for iteration in range(self.num_iter):
            model_output = self.estimator.model(input_sample)
            objective = self.compute_all_classes_lost(model_output)
            objective.backward()  # Compute the gradient form the update of the sample.
            input_sample = input_sample + self.step_size * input_sample.grad.sign()
            input_sample.retain_grad()
        return input_sample.detach().cpu().numpy()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Creating the perturbation x' from the input sample x.
        return: The preprocessed x' and the same y (if exist).
        """
        logger.info("Original dataset size: %d", x.shape[0])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_tensor = torch.Tensor(x).to(self.estimator._device)
        adv_hedged = []
        for sam in x_tensor:
            uniform_noise = torch.rand_like(sam) * 2 - 1  # Creating a uniform noise vector
            uniform_noise *= self.epsilon
            sam += uniform_noise
            sam = torch.tensor(sam).unsqueeze(0).to(device)
            hedged_sample = self.gradient_ascending(sam)
            adv_hedged.append(hedged_sample)
        adv_hedged = np.squeeze(np.array(adv_hedged), axis=1)
        if y:
            return adv_hedged, y
        else:
            return adv_hedged, np.zeros(adv_hedged.shape)

    def _check_params(self) -> None:
        if self.apply_fit:
            raise ValueError("This defence works only on predict.")
