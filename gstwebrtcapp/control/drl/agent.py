import time

from control.agent import Agent, AgentType
from control.controller import Controller
from control.drl.config import DrlConfig
from control.drl.manager import DrlManager
from control.drl.mdp import MDP
from utils.base import LOGGER


class DrlAgent(Agent):
    def __init__(
        self,
        controller: Controller,
        config: DrlConfig,
        mdp: MDP,
        warmup: float = 10.0,
    ) -> None:
        super().__init__(controller)
        self.warmup = warmup
        self.type = AgentType.DRL

        self.manager = DrlManager(config, controller, mdp)

    def run(self, is_load_last_model: bool = False) -> None:
        time.sleep(self.warmup)
        LOGGER.info(f"INFO: DRL Agent warmup {self.warmup} sec is finished, starting...")

        self.manager.reset(is_load_last_model)
        if self.manager.config.mode == "train":
            self.manager.train()
        elif self.manager.config.mode == "eval":
            self.manager.eval()
        else:
            raise Exception(f"Unknown DRL mode {self.manager.config.mode}")

    def stop(self) -> None:
        LOGGER.info("INFO: stopping DRL agent...")
        self.manager.stop()
