import os
import datetime
import argparse
import torch
import numpy as np
import random
from datasets import load_dataset
import transformers
from typing import Optional, Dict, Union, List, Tuple, Any
from torch import nn
import datasets
from packaging import version

import torch.nn.functional as F

from tokenization_sequence import SimpleTokenizer
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorWithPadding, DefaultDataCollator
from transformers.data.data_collator import default_data_collator


class CustomTrainer(transformers.Trainer):

    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class RandomCropDataCollator(DefaultDataCollator):

    def __init__(self, tokenizer, crop_target, time_stamp,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.crop_target = crop_target
        self.time_stamp = time_stamp
        

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:

        crop_target_list = features[self.crop_target]
        time_stamp = features[self.time_stamp]

        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


class MeanTrainer(transformers.Trainer):

    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1}

        emb_i = torch.mean(outputs_0.last_hidden_state, dim=1)
        emb_j = torch.mean(outputs_1.last_hidden_state, dim=1)

    
        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class ThreeTrainer(transformers.Trainer):

    def __init__(self, temperature, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)


        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_2["pooler_output"]

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_2 = torch.sum(loss_partial) / (2 * batch_size)

        loss = loss_1 + self.alpha * loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)


class NegativeSample2LossTrainer(transformers.Trainer):

    def __init__(self, margin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]
        emb_negative = outputs_2["pooler_output"]

        batch_size = emb_i.shape[0]
        # negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)

        loss = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        # sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        # sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        # positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        # nominator = torch.exp(positives / self.temperature)             # 2*bs
        # denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        # loss = torch.sum(loss_partial) / (2 * batch_size)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)
            


class NegativeSample3LossTrainer(transformers.Trainer):

    def __init__(self, temperature, margin, pooling, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin
        self.pooling = pooling

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        # emb_i = outputs_0["pooler_output"]
        # emb_j = outputs_1["pooler_output"]
        # emb_negative = outputs_2["pooler_output"]
        if self.pooling == "cls":
            emb_i = outputs_0["last_hidden_state"][:, 0]
            emb_j = outputs_1["last_hidden_state"][:, 0]
            emb_negative = outputs_2["last_hidden_state"][:, 0]
        else:
            emb_i = torch.mean(outputs_0["last_hidden_state"], dim=1)
            emb_j = torch.mean(outputs_1["last_hidden_state"], dim=1)
            emb_negative = torch.mean(outputs_2["last_hidden_state"], dim=1)

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = loss_1 + loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)
            

# class Triplet_hard_loss(nn.Module):

#     def __init__(self, margin, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.margin = margin
#         self.fct = nn.MarginRankingLoss(margin=margin, reduction="mean")
    
#     def forward(self, z_i, z_j, z_negative):
#         n_dist = torch.cdist(z_i, z_negative)
#         n_dist = torch.min(n_dist, dim=1)[0]

#         p_dist = F.pairwise_distance(z_i, z_j)

#         p_dist = p_dist
#         label = torch.ones_like(p_dist)

#         loss = self.fct(n_dist, p_dist, label)

#         return loss
            
def triplet_hard_loss(z_i, z_j, z_negative, margin):
    n_dist = torch.cdist(z_i, z_negative)
    n_dist = torch.min(n_dist, dim=1)[0]

    p_dist = F.pairwise_distance(z_i, z_j)

    p_dist = p_dist
    label = torch.ones_like(p_dist)

    loss = F.margin_ranking_loss(n_dist, p_dist, label, margin)
    return loss


# def triplet_hard_loss(z_i, z_j, z_negative, margin):
#     n_dist = torch.cdist(z_i, z_negative)
#     n_dist = torch.min(n_dist, dim=1)[0]

#     p_dist = F.pairwise_distance(z_i, z_j)

#     p_dist = p_dist
#     label = torch.ones_like(p_dist)

#     loss = F.margin_ranking_loss(n_dist, p_dist, label, margin)
#     return loss


class NegativeSample3LossTrainerV2(transformers.Trainer):

    def __init__(self, temperature, margin, pooling, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin
        self.pooling = pooling


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        # emb_i = outputs_0["pooler_output"]
        # emb_j = outputs_1["pooler_output"]
        # emb_negative = outputs_2["pooler_output"]
        if self.pooling == "cls":
            emb_i = outputs_0["last_hidden_state"][:, 0]
            emb_j = outputs_1["last_hidden_state"][:, 0]
            emb_negative = outputs_2["last_hidden_state"][:, 0]
        else:
            emb_i = torch.mean(outputs_0["last_hidden_state"], dim=1)
            emb_j = torch.mean(outputs_1["last_hidden_state"], dim=1)
            emb_negative = torch.mean(outputs_2["last_hidden_state"], dim=1)

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        loss_2 = triplet_hard_loss(z_i, z_j, z_negative, self.margin)
        # loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = loss_1 + loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)
            

class NegativeSample3LossTrainerV3(transformers.Trainer):

    def __init__(self, temperature, margin, pooling, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin
        self.pooling = pooling
        self.weight = weight


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        # emb_i = outputs_0["pooler_output"]
        # emb_j = outputs_1["pooler_output"]
        # emb_negative = outputs_2["pooler_output"]
        if self.pooling == "cls":
            emb_i = outputs_0["last_hidden_state"][:, 0]
            emb_j = outputs_1["last_hidden_state"][:, 0]
            emb_negative = outputs_2["last_hidden_state"][:, 0]
        else:
            emb_i = torch.mean(outputs_0["last_hidden_state"], dim=1)
            emb_j = torch.mean(outputs_1["last_hidden_state"], dim=1)
            emb_negative = torch.mean(outputs_2["last_hidden_state"], dim=1)

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        loss_2 = triplet_hard_loss(z_i, z_j, z_negative, self.margin)
        # loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = loss_1 + self.weight*loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class NegativeSample3LossTrainerV3_only_info(transformers.Trainer):

    def __init__(self, temperature, margin, pooling, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin
        self.pooling = pooling
        self.weight = weight


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        # emb_i = outputs_0["pooler_output"]
        # emb_j = outputs_1["pooler_output"]
        # emb_negative = outputs_2["pooler_output"]
        if self.pooling == "cls":
            emb_i = outputs_0["last_hidden_state"][:, 0]
            emb_j = outputs_1["last_hidden_state"][:, 0]
            emb_negative = outputs_2["last_hidden_state"][:, 0]
        else:
            emb_i = torch.mean(outputs_0["last_hidden_state"], dim=1)
            emb_j = torch.mean(outputs_1["last_hidden_state"], dim=1)
            emb_negative = torch.mean(outputs_2["last_hidden_state"], dim=1)

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        # loss_2 = triplet_hard_loss(z_i, z_j, z_negative, self.margin)
        # loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = loss_1 #+ self.weight*loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class NegativeSample3LossTrainerV3_only_triplet(transformers.Trainer):

    def __init__(self, temperature, margin, pooling, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin
        self.pooling = pooling
        self.weight = weight


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_2"], attention_mask=inputs["attention_mask_2"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        # emb_i = outputs_0["pooler_output"]
        # emb_j = outputs_1["pooler_output"]
        # emb_negative = outputs_2["pooler_output"]
        if self.pooling == "cls":
            emb_i = outputs_0["last_hidden_state"][:, 0]
            emb_j = outputs_1["last_hidden_state"][:, 0]
            emb_negative = outputs_2["last_hidden_state"][:, 0]
        else:
            emb_i = torch.mean(outputs_0["last_hidden_state"], dim=1)
            emb_j = torch.mean(outputs_1["last_hidden_state"], dim=1)
            emb_negative = torch.mean(outputs_2["last_hidden_state"], dim=1)

        # batch_size = emb_i.shape[0]
        # negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        # sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        # sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        # positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        # nominator = torch.exp(positives / self.temperature)             # 2*bs
        # denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        # loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        loss_2 = triplet_hard_loss(z_i, z_j, z_negative, self.margin)
        # loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = self.weight*loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class TwoStrategiesTrainer(transformers.Trainer):

    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)

        index = torch.bernoulli(torch.full([inputs["input_ids_0"].shape[0]], 0.5)).bool()
        inputs_1 = inputs["input_ids_1"]
        inputs_2 = inputs["input_ids_2"]
        inputs_1[index] = 0
        inputs_2[~index] = 0

        input_pos = inputs_1 + inputs_2

        attention_mask_1 = inputs["attention_mask_1"]
        attention_mask_2 = inputs["attention_mask_2"]
        attention_mask_1[index] = 0
        attention_mask_2[~index] = 0
        attention_mask_pos = attention_mask_1 + attention_mask_2

        outputs_1 = model(input_ids=input_pos, attention_mask=attention_mask_pos, return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)




class NegativeSample3TwoStrategyLossTrainer(transformers.Trainer):

    def __init__(self, temperature, margin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.margin = margin

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        index = torch.bernoulli(torch.full([inputs["input_ids_0"].shape[0]], 0.5)).bool()
        inputs_1 = inputs["input_ids_1"]
        inputs_2 = inputs["input_ids_2"]
        inputs_1[index] = 0
        inputs_2[~index] = 0

        input_pos = inputs_1 + inputs_2

        attention_mask_1 = inputs["attention_mask_1"]
        attention_mask_2 = inputs["attention_mask_2"]
        attention_mask_1[index] = 0
        attention_mask_2[~index] = 0
        attention_mask_pos = attention_mask_1 + attention_mask_2

        outputs_1 = model(input_ids=input_pos, attention_mask=attention_mask_pos, return_dict=True)
        outputs_2 = model(input_ids=inputs["input_ids_3"], attention_mask=inputs["attention_mask_3"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1, "outputs_2": outputs_2}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]
        emb_negative = outputs_2["pooler_output"]

        batch_size = emb_i.shape[0]
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_negative = F.normalize(emb_negative, dim=1)     # (bs, dim)  --->  (bs, dim)


        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_1 = torch.sum(loss_partial) / (2 * batch_size)

        loss_2 = F.triplet_margin_loss(z_i, z_j, z_negative, margin=self.margin)
        loss = loss_1 + loss_2


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("input_ids_2")
        signature_columns.append("input_ids_3")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("attention_mask_2")
        signature_columns.append("attention_mask_3")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")
        signature_columns.append("token_type_ids_2")
        signature_columns.append("token_type_ids_3")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)



class UserBertTrainer(transformers.Trainer):

    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     weight = inputs.pop("weight")

    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(device=model.device)
    #     loss = weight * loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        # labels = None

        outputs_0 = model(input_ids=inputs["input_ids_0"], attention_mask=inputs["attention_mask_0"], return_dict=True)
        outputs_1 = model(input_ids=inputs["input_ids_1"], attention_mask=inputs["attention_mask_1"], return_dict=True)

        outputs = {"outputs_0": outputs_0, "outputs_1": outputs_1}

        emb_i = outputs_0["pooler_output"]
        emb_j = outputs_1["pooler_output"]

        # emb_i_n = torch.cat([emb_i[-1, :], emb_i[0:-1, :]], dim=0)


        # batch_size = emb_i.shape[0]
        # negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)

        # z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        # z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        # sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        # sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        # positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        # nominator = torch.exp(positives / self.temperature)             # 2*bs
        # denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        # loss = torch.sum(loss_partial) / (2 * batch_size)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs_0[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(outputs)
        # loss = loss * weights.to(loss.device)

        
        return (loss, outputs) if return_outputs else loss



    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns.append("input_ids_0")
        signature_columns.append("input_ids_1")
        signature_columns.append("attention_mask_0")
        signature_columns.append("attention_mask_1")
        signature_columns.append("token_type_ids_0")
        signature_columns.append("token_type_ids_1")


        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            # logger.info(
            #     f"The following columns {dset_description} don't have a corresponding argument in "
            #     f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            #     f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            #     " you can safely ignore this message."
            # )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)




    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            
            labels = None

            with torch.no_grad():              
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()


            if prediction_loss_only:
                return (loss, None, None)

