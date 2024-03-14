'''
Test main functionalities by replacing the default one with one of the desired coroutine presented above in the main() endpoint.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>
'''

import asyncio
import aioprocessing
from aioprocessing import AioProcess, AioManager
from apps.app import GstWebRTCAppConfig
from apps.ahoyapp.connector import AhoyConnector
from apps.pipelines import DEFAULT_BIN_CUDA_PIPELINE, DEFAULT_SINK_PIPELINE
from apps.sinkapp.connector import SinkConnector
from control.controller import Controller
from control.drl.agent import FedAgent
from control.drl.config import FedConfig
from control.drl.mdp import ViewerMDP
from control.drl.mdp import ViewerSeqMDP
from message.broker import MosquittoBroker
from message.client import MqttConfig
from network.controller import NetworkController
from utils.base import LOGGER
from apps.pipelines import DEFAULT_H265_IN_WEBRTCBIN_H264_OUT_PIPELINE

try:
    import uvloop
except ImportError:
    uvloop = None

AHOY_DIRECTOR_URL = "https://devdirex.wavelab.addix.net/api/v2/feed/attach/"
API_KEY = "1f3ca3c3c6580a07fca62e18c2d6f325802b681a"
VIDEO_SOURCE = "rtsp://admin:@kiel.ins.informatik.uni-kiel.de:554/h264Preview_01_sub"

def average_weights(weights_list):
    weightResult = {}
    for key in weights_list[0]:
        weightResult[key] = sum(weights[key] for weights in weights_list) / len(weights_list)
    return weightResult  


def create_workers(num_workers, result_queue, update_queue, update_freq):
    workers = []
    for i in range(num_workers):
        if i == 0:
            p = aioprocessing.AioProcess(target=test_fed_start, args=(f"feed_{i}", result_queue, update_queue, update_freq, True))
            p.start()
            workers.append(p)
        else:
            p = aioprocessing.AioProcess(target=test_fed_start, args=(f"feed_{i}", result_queue, update_queue, update_freq, False))
            p.start()
            workers.append(p)
    return workers


def test_fed_start(feed_name, result_queue, update_queue, update_freq, isLogging):
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(test_fed(feed_name, result_queue, update_queue, update_freq, isLogging))

async def test_fed(feed_name, result_queue, update_queue, update_freq, isLogging):
    # run it to test drl agent
    try:
        mqtt_cfg = MqttConfig(id="", broker_port=1884)
        episodes = 200
        episode_length = 50
        stats_update_interval = 1.0

        app_cfg = GstWebRTCAppConfig(
            video_url=VIDEO_SOURCE, 
            pipeline_str=DEFAULT_SINK_PIPELINE, 
            codec="h264", 
            bitrate=2000, 
            gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
            )
        
        callbacksToUse = ['print_step', 'federated']
        verbosity = 1
        if(isLogging):
            callbacksToUse = ['print_step', 'federated', 'save_step', 'save_model']
            verbosity = 2

        agent = FedAgent(
            config=FedConfig(
                mode="train",
                model_name="sac",
                episodes=episodes,
                episode_length=episode_length,
                state_update_interval=stats_update_interval,
                result_queue=result_queue,
                update_queue=update_queue,
                update_freq=update_freq,
                hyperparams_cfg={
                    "policy": "MultiInputPolicy",
                    "batch_size": 128,
                    "ent_coef": "auto",
                    "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [256, 256]},
                },
                callbacks=callbacksToUse,
                save_model_path="./fedModels",
                save_log_path="./fedLogs",
                verbose=verbosity,
            ),
            controller=Controller(),
            mdp=ViewerMDP(
                reward_function_name="qoe_fed",
                episode_length=episode_length,
                constants={
                    "MAX_BITRATE_STREAM_MBPS": 10,
                    "MAX_BANDWIDTH_MBPS": app_cfg.gcc_settings["max-bitrate"] / 1000000,
                    "MIN_BANDWIDTH_MBPS": app_cfg.gcc_settings["min-bitrate"] / 1000000,
                },
            ),
            mqtt_config = mqtt_cfg,
        )

        conn = SinkConnector(
            pipeline_config=app_cfg,
            agent=agent,
            mqtt_config=mqtt_cfg,
            NetworkController=NetworkController(),
        )

        await conn.connect_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


def update_loop(num_workers, result_queue, update_queue):
    while True:
        # Wait until the result queue is full
        while not result_queue.full():
            pass
        # Get all the weights from the result queue
        weights = []
        for i in range(num_workers):
            weights.append(result_queue.get())
    
        # Average the weights and fill the update queue with the averaged weights
        averaged_weights = average_weights(weights)
        for i in range(num_workers):
            update_queue.put(averaged_weights)
            

if __name__ == "__main__":
    # Create n federated workers
    num_workers = 2
    with aioprocessing.AioManager() as manager:
        result_queue = manager.AioQueue(maxsize=num_workers)
        update_queue = manager.AioQueue(maxsize=num_workers)
        workers = create_workers(num_workers, result_queue, update_queue, update_freq=10)
        # Start the weight update loop
        update_loop(num_workers, result_queue, update_queue)  
    
    
    
    
    
    
