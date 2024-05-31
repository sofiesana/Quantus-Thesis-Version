"""This module contains the implementation of the Iterative Removal of Features metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import sys
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from quantus.functions.perturb_func import baseline_replacement_by_indices
from quantus.helpers import asserts, utils, warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class IROF(Metric[List[float]]):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    Assumptions:
        - The original metric definition relies on image-segmentation functionality. Therefore, only apply the
        metric to 3-dimensional (image) data. To extend the applicablity to other data domains,
        adjustments to the current implementation might be necessary.

    References:
        1) Laura Rieger and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "IROF"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        segmentation_method: str = "slic",
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: str = "mean",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        class_category: int = None,
        num_classes: int = 1,
        class_name: str = "Class",
        task: str = "seg",
        **kwargs,
    ):
        """
        Parameters
        ----------
        segmentation_method: string
            Image segmentation method:'slic' or 'felzenszwalb', default="slic".
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="mean".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        # Save metric-specific attributes.
        self.num_classes = num_classes
        self.class_category = class_category
        self. class_name = class_name
        self.task = task
        self.segmentation_method = segmentation_method
        self.nr_channels = None
        self.perturb_func = make_perturb_func(
            perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline
        )

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated"
                    " hyperparameters), also, IROF only works with image data"
                ),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            **kwargs,
        )
    
    def get_y_pred_sn(self, model, x_input, y):
        y_pred = model.predict(x_input)

        # print("y_pred:", y_pred)
        # print("y_pred shape:", y_pred.shape)

        # print("y:", y)
        # print("y shape:", y.shape)

        # Convert numpy arrays to PyTorch tensors
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Reshape tensors to (batch_size, n*m, 3) to facilitate cosine similarity element-wise
        batch_size, channels, n, m = y_pred_tensor.shape
        y_pred_tensor = y_pred_tensor.permute(0, 2, 3, 1).reshape(batch_size, n * m, channels)
        y = torch.unsqueeze(y, 0).permute(0, 2, 3, 1).reshape(batch_size, n * m, channels)
        print("y shape:", y.shape)

        # Define cosine similarity function
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        # Compute cosine similarity
        cosine_similarities = cos(y_pred_tensor.to(y.device), y)

        # Reshape back to original shape (batch_size, n, m)
        # not needed if only mean is needed
        cosine_similarities = cosine_similarities.view(batch_size, n, m)

        # try this later:
        # rescaled_cosine_similarities = (cosine_similarities + 1)

        np_cosine_similarity = cosine_similarities.detach().cpu().numpy()

        mean_cs = np.mean(np_cosine_similarity)
        print("cosine_similarities:", cosine_similarities)
        print("mean:", mean_cs)

        return mean_cs

    def get_y_pred(self, model, x_input, y):
        y_pred = model.predict(x_input)
        # y_pred = torch.from_numpy(y_pred)
        # y_pred = F.softmax(y_pred, dim = 1)

        # checking
        # reshape predictions and flatten
        # y_pred_reshaped = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, 40)
        
        y_pred_reshaped = np.transpose(y_pred, (0, 2, 3, 1)).reshape(-1, 40)
    
        # reshape labels and flatten
        new_shape = y_pred.shape[-2:]
        # y_resized = F.interpolate(torch.unsqueeze(y, 0), size=new_shape)
        y_resized = y
        # y = y_resized.permute(0, 2, 3, 1).contiguous().view(-1)
        y = y.contiguous().view(-1)
        y = y.long()
        y = y.cpu().numpy()

        # filter to only keep pixels of class of interest
        class_category_mask = (y == self.class_category)
        filtered_pred = y_pred_reshaped[torch.arange(y.shape[0]), y]
        y_pred = filtered_pred[class_category_mask]

        # print("y_pred just masked:", y_pred_reshaped[torch.arange(y.shape[0]), class_category_mask])
        # y_pred = y_pred.cpu().numpy()
        
        # get average score
        y_pred = np.mean(y_pred)
        
        return y_pred

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: torch.Tensor,
        y: np.ndarray,
        a: np.ndarray,
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x: torch.Tensor
            The input to be evaluated on an instance-basis. Already on GPU.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        Returns
        -------
        float
            The evaluation results.
        """
        if self.class_category not in y:
            print(self.class_name + ' does not exist in this image')
            return None, None

        # Predict on x.        
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = self.get_y_pred(model, x_input, y)
        # print("############################### ORIGINAL Y PRED:", y_pred)

        # Move x to CPU and convert to NumPy array for segmentation
        cpu_numpy_x = x.cpu().numpy()

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(cpu_numpy_x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []
        x_prev_perturbed = x

        for i_ix, s_ix in enumerate(s_indices):
            # Move x_prev_perturbed to CPU and convert to NumPy array for perturbation
            x_prev_perturbed_cpu = x_prev_perturbed.cpu().numpy()

            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x_prev_perturbed_cpu,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )
            warn.warn_perturbation_caused_no_change(
                x=x_prev_perturbed_cpu, x_perturbed=x_perturbed
            )

            # Convert x_perturbed back to a PyTorch tensor and move to GPU
            x_perturbed_tensor = torch.from_numpy(x_perturbed).to(x.device)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed_tensor, x_perturbed_tensor.shape, channel_first=True)
            y_pred_perturb = self.get_y_pred(model, x_input, y)
            # print("############################### Y PRED PERTURBED:", y_pred_perturb)

            # Normalize the scores to be within range [0, 1].
            preds.append(float(y_pred_perturb / y_pred))
            x_prev_perturbed = x_perturbed_tensor

        # Calculate the area over the curve (AOC) score.
        aoc = len(preds) - utils.calculate_auc(np.array(preds))

        return aoc, preds
    
    def evaluate_instance_sn(
        self,
        model: ModelInterface,
        x: torch.Tensor,
        y: np.ndarray,
        a: np.ndarray
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x: torch.Tensor
            The input to be evaluated on an instance-basis. Already on GPU.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        Returns
        -------
        float
            The evaluation results.
        """
        # Predict on x.        
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = self.get_y_pred_sn(model, x_input, y)
        # print("############################### ORIGINAL Y PRED:", y_pred)

        # Move x to CPU and convert to NumPy array for segmentation
        cpu_numpy_x = x.cpu().numpy()

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(cpu_numpy_x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = len(np.unique(segments))
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []
        x_prev_perturbed = x

        for i_ix, s_ix in enumerate(s_indices):
            # Move x_prev_perturbed to CPU and convert to NumPy array for perturbation
            x_prev_perturbed_cpu = x_prev_perturbed.cpu().numpy()

            # Perturb input by indices of attributions.
            a_ix = np.nonzero((segments == s_ix).flatten())[0]

            x_perturbed = self.perturb_func(
                arr=x_prev_perturbed_cpu,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )
            warn.warn_perturbation_caused_no_change(
                x=x_prev_perturbed_cpu, x_perturbed=x_perturbed
            )

            # Convert x_perturbed back to a PyTorch tensor and move to GPU
            x_perturbed_tensor = torch.from_numpy(x_perturbed).to(x.device)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed_tensor, x_perturbed_tensor.shape, channel_first=True)
            y_pred_perturb = self.get_y_pred_sn(model, x_input, y)
            # print("############################### Y PRED PERTURBED:", y_pred_perturb)

            # Normalize the scores to be within range [0, 1].
            preds.append(float(y_pred_perturb / y_pred))
            x_prev_perturbed = x_perturbed_tensor

        # Calculate the area over the curve (AOC) score.
        aoc = len(preds) - utils.calculate_auc(np.array(preds))

        return aoc, preds

    def custom_preprocess(
        self,
        x_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        # Infer number of input channels.
        self.nr_channels = x_batch.shape[1]

    @property
    def get_aoc_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return np.mean(self.evaluation_scores)

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> List[float]:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            The evaluation results.
        """
        scores = []
        histories = []


        for x, y, a in zip(x_batch, y_batch, a_batch):
            if self.task == "seg":
                score, history = self.evaluate_instance(model=model, x=x, y=y, a=a)
            elif self.task == "sn":
                score, history = self.evaluate_instance_sn(model=model, x=x, y=y, a=a)
            if score is not None:
                scores.append(score)
                histories.append(history)
        return scores, histories

