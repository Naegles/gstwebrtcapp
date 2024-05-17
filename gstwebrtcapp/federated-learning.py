import asyncio
from multiprocessing import Process, Manager, Queue, JoinableQueue
from aioprocessing import AioManager
from queue import Empty, Full
from apps.app import GstWebRTCAppConfig
from apps.pipelines import DEFAULT_SINK_PIPELINE
from apps.sinkapp.connector import SinkConnector
from control.drl.agent import FedAgent
from control.drl.config import FedConfig
from control.drl.mdp import ViewerSeqMDP
from message.client import MqttConfig
from network.controller import NetworkController
from utils.base import LOGGER
from typing import Callable

try:
    import uvloop
except ImportError:
    uvloop = None

VIDEO_SOURCE = "rtsp://192.168.178.30"
PORT_START = 57883


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


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
        mqtt_cfg = MqttConfig(feed_name=feed_name, broker_port=1884)
        episodes = 10
        episode_length = 50
        stats_update_interval = 2.0

        app_cfg = GstWebRTCAppConfig( 
            pipeline_str=DEFAULT_SINK_PIPELINE, 
            video_url=VIDEO_SOURCE + ':' + str(port),
            codec="h264", 
            bitrate=6000, 
            gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
            is_debug=False
        )
        
        # Only output and save logs of one of the agents
        callbacksToUse = ['federated']
        verbosity = 0
        if(isMain):
            callbacksToUse = ['print_step', 'federated', 'save_step', 'save_model']
            verbosity = 2

        schedule = linear_schedule(0.0007)

        agent = FedAgent(
            config=FedConfig(
                mode="eval",
                model_file="fedModelsReward/rewardRate = 0.05 - new - bitrate15000/drl_model_5000_steps.zip",
                deterministic=True,
                model_name="sac",
                episodes=episodes,
                episode_length=episode_length,
                state_update_interval=stats_update_interval,
                result_queue=result_queue,
                update_queue=update_queue,
                update_freq=update_freq,
                hyperparams_cfg={
                    "policy": "MultiInputPolicy",
                    "gamma" : 0.999,
                    "learning_rate" : schedule,
                    "batch_size": 512,
                    "tau" : 0.02,
                    "ent_coef": 0.1,
                    "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [256, 256]},
                    "learning_starts": 10,
                    "seed" : seed,
                },
                callbacks=callbacksToUse,
                save_log_path="./fedLogsRewardEval",
                # save_model_path="./fedModelsReward",
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
            network_controller = NetworkController(gt_bandwidth=10.0, interval=15.0, interface="lo")
            network_controller.generate_rules(100, [0.5, 0.25, 0.25])
            network_controller.rules = ['rate 9.155231310871939Mbps', 'rate 8.447925813491521Mbps', 'rate 1.714193522212855Mbps', 'rate 7.394084162247659Mbps', 'rate 9.7701340312009Mbps', 'rate 1.3496190427369463Mbps', 'rate 0.7629224857085939Mbps', 'rate 4.9042717558844355Mbps', 'rate 1.125849491839105Mbps', 'rate 1.334029703012135Mbps', 'rate 5.640485807699355Mbps', 'rate 1.4255804498932054Mbps', 'rate 9.633731303368744Mbps', 'rate 8.645886435762382Mbps', 'rate 1.0470432228761348Mbps', 'rate 4.500341158971216Mbps', 'rate 9.202918211183508Mbps', 'rate 4.859650777984523Mbps', 'rate 8.11577207726139Mbps', 'rate 0.961251259333064Mbps', 'rate 8.523394754217762Mbps', 'rate 8.902576944255207Mbps', 'rate 8.879048252761137Mbps', 'rate 9.468335477311658Mbps', 'rate 0.6180027331890625Mbps', 'rate 1.2014965519835725Mbps', 'rate 1.824192025736446Mbps', 'rate 0.5370252264224726Mbps', 'rate 8.243286314880372Mbps', 'rate 9.234984798397532Mbps', 'rate 0.7731001383557897Mbps', 'rate 1.6740967295152853Mbps', 'rate 1.5779765698276025Mbps', 'rate 6.603350815571583Mbps', 'rate 9.191318447558348Mbps', 'rate 7.3094969395497Mbps', 'rate 9.325269051747199Mbps', 'rate 6.520659845058197Mbps', 'rate 8.091250428473861Mbps', 'rate 4.424689227455319Mbps', 'rate 7.389684550647091Mbps', 'rate 8.027691181116248Mbps', 'rate 6.824447713009686Mbps', 'rate 7.276015887504078Mbps', 'rate 8.56002069322357Mbps', 'rate 6.497927121889034Mbps', 'rate 8.284979471056678Mbps', 'rate 9.176361720285668Mbps', 'rate 1.0960519903183046Mbps', 'rate 0.6950032652007753Mbps', 'rate 8.77998717217318Mbps', 'rate 8.343515107662936Mbps', 'rate 8.108406753725887Mbps', 'rate 0.7059096906378777Mbps', 'rate 7.451099656435201Mbps', 'rate 8.39703962437738Mbps', 'rate 1.8041421719419626Mbps', 'rate 1.220078135958626Mbps', 'rate 9.68335998618313Mbps', 'rate 1.2654287293127255Mbps', 'rate 9.353478598085047Mbps', 'rate 1.3710570297810745Mbps', 'rate 8.737122694276206Mbps', 'rate 6.943000345855733Mbps', 'rate 0.6994030211814091Mbps', 'rate 5.461007201403179Mbps', 'rate 1.7968734979826535Mbps', 'rate 8.34723071008508Mbps', 'rate 9.293431080007334Mbps', 'rate 1.1863288359533217Mbps', 'rate 8.61287282497544Mbps', 'rate 9.56282447490503Mbps', 'rate 6.56243770108955Mbps', 'rate 0.8949833000057321Mbps', 'rate 5.159338923745569Mbps', 'rate 9.086364326460469Mbps', 'rate 8.609872299996391Mbps', 'rate 4.111921310903051Mbps', 'rate 8.607049197994858Mbps', 'rate 1.9310608144952317Mbps', 'rate 9.77904283470087Mbps', 'rate 5.451394152350728Mbps', 'rate 6.189988425383209Mbps', 'rate 0.5583341348926684Mbps', 'rate 8.312446060434102Mbps', 'rate 1.8745589520815389Mbps', 'rate 4.807145206840568Mbps', 'rate 8.787068094490946Mbps', 'rate 6.462319336265455Mbps', 'rate 1.547552159841859Mbps', 'rate 8.502239345632235Mbps', 'rate 1.479593036246653Mbps', 'rate 9.109137579074833Mbps', 'rate 1.662593139889464Mbps', 'rate 1.6790011343911597Mbps', 'rate 6.47481966008876Mbps', 'rate 7.029524203404427Mbps', 'rate 1.840561156407922Mbps', 'rate 0.756696846466147Mbps', 'rate 4.7424242452828445Mbps']


        
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
                    weights.append(result_queue.get(180))
                    result_queue.task_done()
                except Empty:
                    LOGGER.info("ERROR: Timeout while update_loop waiting for weight update from agents")
                    
        # Average the weights and fill the update queue with the averaged weights
        averaged_weights = average_weights(weights)
        for i in range(num_workers):
            try:
                update_queue.put(averaged_weights, timeout=180)
            except Full:
                LOGGER.info("ERROR: Timeout while update_loop waiting for update queue to be free")
        update_queue.join()           


if __name__ == "__main__":
    # Create n federated workers
    num_workers = 1
    with Manager() as manager:
        result_queue = manager.JoinableQueue(maxsize=num_workers)
        update_queue = manager.JoinableQueue(maxsize=num_workers)
        workers = create_workers(num_workers, result_queue, update_queue, update_freq=5)
        # Start the weight update loop
        update_loop(num_workers, result_queue, update_queue)  

    """ networkController = network_controller = NetworkController(gt_bandwidth=10.0, interval=15.0, interface="lo")
    network_controller.generate_rules(100, [0.4, 0.3, 0.3])
    print(network_controller.rules) """