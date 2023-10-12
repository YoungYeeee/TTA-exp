# -*- coding: utf-8 -*-
import copy
from typing import Any, Dict
import ttab.utils.step_loss_grad as loss_grad
from ttab.model_selection.base_selection import BaseSelection


class LastIterate(BaseSelection):
    """Naively return the model generated from the last iterate of adaptation."""

    def __init__(self, meta_conf, model_adaptation_method):
        super().__init__(meta_conf, model_adaptation_method)
        self.meta_conf.step = 0
    def initialize(self):
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.optimal_state = None

    def clean_up(self):
        self.optimal_state = None

    def save_state(self, state, current_batch):
        self.optimal_state = state
        self.meta_conf.step += 1
        loss_grad.saveAsCSV(self.meta_conf,state,current_batch)
        

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        return self.optimal_state

    @property
    def name(self):
        return "last_iterate"
