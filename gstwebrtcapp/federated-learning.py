'''
Test main functionalities by replacing the default one with one of the desired coroutine presented above in the main() endpoint.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>
'''

import asyncio
import aioprocessing

from apps.app import GstWebRTCAppConfig
from apps.ahoyapp.connector import AhoyConnector
from apps.pipelines import DEFAULT_BIN_CUDA_PIPELINE, DEFAULT_SINK_PIPELINE
from control.controller import Controller
from control.drl.agent import FedAgent
from control.drl.config import FedConfig
from control.drl.mdp import ViewerMDP
from utils.base import LOGGER
from apps.pipelines import DEFAULT_H265_IN_WEBRTCBIN_H264_OUT_PIPELINE

try:
    import uvloop
except ImportError:
    uvloop = None

AHOY_DIRECTOR_URL = "https://devdirex.wavelab.addix.net/api/v2/feed/attach/"
API_KEY = "1f3ca3c3c6580a07fca62e18c2d6f325802b681a"
VIDEO_SOURCE = "rtsp://admin:@10.10.10.125:554/h265Preview_01_main"

def average_weights(weights_list):
    weightResult = {}
    for key in weights_list[0]:
        weightResult[key] = sum(weights[key] for weights in weights_list) / len(weights_list)
    return weightResult

    
def test_fed_start(feed_name, result_queue, update_queue, update_freq):
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(test_fed(feed_name, result_queue, update_queue, update_freq))

async def test_fed(feed_name, result_queue, update_queue, update_freq):
    # run it to test drl agent
    try:
        episodes = 10
        episode_length = 50
        stats_update_interval = 3.0

        app_cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE, pipeline_str=DEFAULT_H265_IN_WEBRTCBIN_H264_OUT_PIPELINE)

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
                callbacks=['print_step', 'federated'],
                save_model_path="./models",
                save_log_path="./logs",
                verbose=2,
            ),
            controller=Controller(),
            mdp=ViewerMDP(
                reward_function_name="qoe_ahoy",
                episode_length=episode_length,
                constants={"MAX_BITRATE_STREAM_MBPS": 6},  # Ahoy fixes the max bitrate to 6 Mbps in SDP
            ),
        )

        conn = AhoyConnector(
            pipeline_config=app_cfg,
            agent=agent,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name=feed_name,
            stats_update_interval=stats_update_interval,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


def update_loop(number_of_updates, num_workers, result_queue, update_queue):
    for i in range(number_of_updates):
        # Wait until the result queue is full
        while not result_queue.full():
            pass
        # Get all the weights from the result queue
        weights = []
        for i in range(num_workers):
            weights.append(result_queue.get())   
        print(weights)
    
    # Average the weights and fill the update queue with the averaged weights
    averaged_weights = average_weights(weights)
    for i in range(num_workers):
        update_queue.put(averaged_weights)


def create_workers(num_workers, result_queue, update_queue, update_freq):
    workers = []
    for i in range(num_workers):
        p = aioprocessing.AioProcess(target=test_fed_start, args=(f"feed_{i}", result_queue, update_queue, update_freq))
        p.start()
        workers.append(p)
    return workers


if __name__ == "__main__":
    # Create n federated workers
    num_workers = 2
    result_queue = aioprocessing.Queue(maxsize=num_workers)
    update_queue = aioprocessing.Queue(maxsize=num_workers)
    workers = create_workers(num_workers, result_queue, update_queue, update_freq=10)
    
    # Start the weight update loop
    update_loop(10, num_workers, result_queue, update_queue)
    
    
    
    
    
    
    
