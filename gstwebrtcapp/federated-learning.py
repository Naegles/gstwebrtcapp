'''
Test main functionalities by replacing the default one with one of the desired coroutine presented above in the main() endpoint.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>
'''

import asyncio
from multiprocessing import Process, Manager, Queue
from aioprocessing import AioManager
from queue import Empty, Full
from apps.app import GstWebRTCAppConfig
from apps.pipelines import DEFAULT_SINK_PIPELINE
from apps.sinkapp.connector import SinkConnector
from control.drl.agent import FedAgent
from control.drl.config import FedConfig
from control.drl.mdp import ViewerMDP
from control.drl.mdp import ViewerSeqMDP
from message.client import MqttConfig
from network.controller import NetworkController
from utils.base import LOGGER

try:
    import uvloop
except ImportError:
    uvloop = None

VIDEO_SOURCE = "rtsp://192.168.178.30"
PORT_START = 57883

def average_weights(weights_list):
    weightResult = {}
    for key in weights_list[0]:
        weightResult[key] = sum(weights[key] for weights in weights_list) / len(weights_list)
    return weightResult  


def create_workers(num_workers, result_queue, update_queue, update_freq):
    workers = []
    for i in range(num_workers):
        port = PORT_START + i
        if i == 0:
            p = Process(target=test_fed_start, args=(f"feed_{i}", i, result_queue, update_queue, update_freq, True, port))
            p.start()
            workers.append(p)
        else:
            p = Process(target=test_fed_start, args=(f"feed_{i}", i, result_queue, update_queue, update_freq, False, port))
            p.start()
            workers.append(p)
    return workers


def test_fed_start(feed_name, seed, result_queue, update_queue, update_freq, isMain, port):
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(test_fed(feed_name, seed, result_queue, update_queue, update_freq, isMain, port))

async def test_fed(feed_name, seed, result_queue, update_queue, update_freq, isMain, port):
    # run it to test fed agent
    try:
        mqtt_cfg = MqttConfig(feed_name=feed_name, broker_port=1883)
        episodes = 100
        episode_length = 50
        stats_update_interval = 2.0

        app_cfg = GstWebRTCAppConfig( 
            pipeline_str=DEFAULT_SINK_PIPELINE, 
            video_url=VIDEO_SOURCE + ':' + str(port),
            codec="h264", 
            bitrate=2000, 
            gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
            is_debug=False
        )
        
        # Only output and save logs of one of the agents
        callbacksToUse = ['federated']
        verbosity = 0
        if(isMain):
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
                    "ent_coef": "auto_0.1",
                    "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [256, 256]},
                    "learning_starts": 10,
                    "seed" : seed,
                },
                callbacks=callbacksToUse,
                save_model_path="./fedModels",
                save_log_path="./fedLogs",
                verbose=verbosity,
            ),
            mdp=ViewerSeqMDP(
                reward_function_name="qoe_ahoy_seq",
                episode_length=episode_length,
                constants={
                    "MAX_BITRATE_STREAM_MBPS": 10,
                    "MAX_BANDWIDTH_MBPS": app_cfg.gcc_settings["max-bitrate"] / 1000000,
                    "MIN_BANDWIDTH_MBPS": app_cfg.gcc_settings["min-bitrate"] / 1000000,
                },
            ),
            mqtt_config = mqtt_cfg,
        )

        # Make sure there is only one network controller
        network_controller = None
        if isMain:
            network_controller = NetworkController(gt_bandwidth=10.0, interval=30.0, interface="lo")
            network_controller.generate_rules(100, [0.7, 0.2, 0.1])

        
        conn = SinkConnector(
            pipeline_config=app_cfg,
            agent=agent,
            mqtt_config=mqtt_cfg,
            network_controller=network_controller,
            feed_name=feed_name
        )

        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


def update_loop(num_workers, result_queue, update_queue):
    while True:
        # Get all the weights from the result queue
        weights = []
        for i in range(num_workers):
                try:
                    weights.append(result_queue.get(timeout=180))
                except Empty:
                    LOGGER.info("ERROR: Timeout while update_loop waiting for weight update from agents")
                    pass
                    
        # Average the weights and fill the update queue with the averaged weights
        averaged_weights = average_weights(weights)
        for i in range(num_workers):
            try:
                update_queue.put(averaged_weights, timeout=30)
            except Full:
                LOGGER.info("ERROR: Timeout while update_loop waiting for update queue to be free") 
                pass           

if __name__ == "__main__":
    # Create n federated workers
    num_workers = 4
    with Manager() as manager:
        result_queue = manager.Queue(maxsize=num_workers)
        update_queue = manager.Queue(maxsize=num_workers)
        workers = create_workers(num_workers, result_queue, update_queue, update_freq=10)
        # Start the weight update loop
        update_loop(num_workers, result_queue, update_queue)  
        
    
    
    
    
    
