import asyncio
from multiprocessing import Process, Manager
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
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    uvloop = None

# Global configuration variables
VIDEO_SOURCE = "rtsp://192.168.178.30"
PORT_START = 57884

NUM_WORKERS = 1
EPISODES = 5
EPISODE_LENGTH = 50
STATE_UPDATE_INTERVAL = 2
WEIGHT_AGGREGATION_FREQUENCY = 5
DETERMINISTIC = True

MODEL_FILE = f"fedModelsReward/last_default.zip"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Creates a linear learning rate schedule.
    
    Args:
        initial_value: The initial learning rate.

    Returns:
        A function that computes the current learning rate based on the remaining progress.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule

def average_weights(weights_list):
    """
    Averages weights across multiple training instances.

    Args:
        weights_list: List of dictionaries containing parameter weights.

    Returns:
        Dictionary of averaged weights.
    """
    weight_result = {}
    for key in weights_list[0]:
        weight_result[key] = sum(weights[key] for weights in weights_list) / len(weights_list)
    return weight_result

def create_workers(num_workers, result_queue, update_queue, update_freq):
    """
    Creates and starts worker processes.

    Args:
        num_workers: Number of worker processes to create.
        result_queue: Queue to collect results from workers.
        update_queue: Queue to push updates to workers.
        update_freq: Frequency of updates.

    Returns:
        List of worker processes.
    """
    workers = []
    for i in range(num_workers):
        port = PORT_START + i
        is_main = i == 0
        args = (f"feed_{i}", i, result_queue, update_queue, update_freq, is_main, port)
        p = Process(target=test_fed_start, args=args)
        p.start()
        workers.append(p)
    return workers

def test_fed_start(feed_name, seed, result_queue, update_queue, update_freq, is_main, port):
    """
    Starts the federated learning process for a worker.
    """
    asyncio.run(test_fed(feed_name, seed, result_queue, update_queue, update_freq, is_main, port))

async def test_fed(feed_name, seed, result_queue, update_queue, update_freq, is_main, port):
    """
    Asynchronous task to simulate federated learning for a worker.
    """
    mqtt_cfg = MqttConfig(feed_name=feed_name, broker_port=1883)
    app_cfg = GstWebRTCAppConfig(
        pipeline_str=DEFAULT_SINK_PIPELINE,
        video_url=f"{VIDEO_SOURCE}:{port}",
        codec="h264",
        bitrate=6000,
        gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
        is_debug=False
    )

    callbacks_to_use = ['print_step', 'federated', 'save_step', 'save_model'] if is_main else ['federated']
    verbosity = 2 if is_main else 0

    agent = FedAgent(
        config=FedConfig(
            mode="eval",
            model_file=MODEL_FILE,
            deterministic=DETERMINISTIC,
            model_name="sac",
            episodes=EPISODES,
            episode_length=EPISODE_LENGTH,
            state_update_interval=STATE_UPDATE_INTERVAL,
            result_queue=result_queue,
            update_queue=update_queue,
            update_freq=update_freq,
            hyperparams_cfg={
                "policy": "MultiInputPolicy",
                "gamma": 0.98,
                "learning_rate": 0.0007,
                "buffer_size": 300000,
                "batch_size": 128,
                "tau": 0.02,
                "train_freq": 64,
                "gradient_steps": 64,
                "ent_coef": "auto",
                "use_sde": True,
                "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [512, 256, 256]},
                "learning_starts": 100,
                "seed": seed,
            },
            callbacks=callbacks_to_use,
            save_log_path="./fedLogsRewardEval",
            verbose=verbosity,
        ),
        mdp=ViewerSeqMDP(
            reward_function_name="qoe_ahoy_seq",
            episode_length=EPISODE_LENGTH,
            constants={
                "MAX_BITRATE_STREAM_MBPS": 10,
                "MAX_BANDWIDTH_MBPS": app_cfg.gcc_settings["max-bitrate"] / 1000000,
                "MIN_BANDWIDTH_MBPS": app_cfg.gcc_settings["min-bitrate"] / 1000000,
            },
        ),
        mqtt_config=mqtt_cfg,
    )

    # Make sure there is only one network controller
    network_controller = None
    if is_main:
        network_controller = NetworkController(gt_bandwidth=10.0, interval=30.0, interface="lo")
        network_controller.generate_rules(100, [0.7, 0.2, 0.1])

        # New Network Rules
        network_controller.rules = ['rate 10.795984749727Mbps', 'rate 11.393509829463Mbps', 'rate 10.749737799605Mbps', 'rate 10.366684073629Mbps', 'rate 8.830054987626Mbps', 'rate 3.954909283673Mbps', 'rate 10.667162173637Mbps', 'rate 10.839174095166Mbps', 'rate 10.313910955653Mbps', 'rate 10.902768242888Mbps', 'rate 11.858967476581Mbps', 'rate 3.652151451236Mbps', 'rate 3.397560482896Mbps', 'rate 10.867875691681Mbps', 'rate 3.462503489973Mbps', 'rate 6.825626994975Mbps', 'rate 11.530672988981Mbps', 'rate 11.457762597168Mbps', 'rate 3.049078521788Mbps', 'rate 10.739065774439Mbps', 'rate 3.889903666907Mbps', 'rate 9.367391080155Mbps', 'rate 3.824306080930Mbps', 'rate 3.192685797840Mbps', 'rate 3.186041716005Mbps', 'rate 10.403196262721Mbps', 'rate 3.703561894298Mbps', 'rate 3.302443442075Mbps', 'rate 10.201513707428Mbps', 'rate 3.410814029226Mbps', 'rate 10.558720176662Mbps', 'rate 3.021790831755Mbps', 'rate 3.077485365821Mbps', 'rate 3.971277373948Mbps', 'rate 11.235678956229Mbps', 'rate 10.982674951671Mbps', 'rate 11.787526849219Mbps', 'rate 9.507232370024Mbps', 'rate 3.401523256595Mbps', 'rate 10.654496096526Mbps', 'rate 3.340767698688Mbps', 'rate 3.628600390916Mbps', 'rate 11.684440600925Mbps', 'rate 11.557353285958Mbps', 'rate 10.254508721843Mbps', 'rate 10.054247242836Mbps', 'rate 6.832785209820Mbps', 'rate 3.572438930118Mbps', 'rate 3.826550951148Mbps', 'rate 3.439976290823Mbps', 'rate 3.916218127180Mbps', 'rate 8.391753168314Mbps', 'rate 10.883194046604Mbps', 'rate 3.202675475322Mbps', 'rate 10.293378041323Mbps', 'rate 11.381967540470Mbps', 'rate 11.767159116774Mbps', 'rate 10.098097858454Mbps', 'rate 8.659052258404Mbps', 'rate 6.725635145466Mbps', 'rate 7.254192568661Mbps', 'rate 10.065587353074Mbps', 'rate 11.703997866125Mbps', 'rate 9.125193226739Mbps', 'rate 6.392147723330Mbps', 'rate 3.924592523093Mbps', 'rate 10.501935980495Mbps', 'rate 3.749493367220Mbps', 'rate 3.688084212776Mbps', 'rate 8.648800539511Mbps', 'rate 6.119569963433Mbps', 'rate 10.455362317100Mbps', 'rate 3.430098552524Mbps', 'rate 3.078501710999Mbps', 'rate 8.708403239637Mbps', 'rate 3.188809870844Mbps', 'rate 6.943017781430Mbps', 'rate 11.827810347715Mbps', 'rate 9.788337357613Mbps', 'rate 10.624014099682Mbps', 'rate 10.463318483249Mbps', 'rate 7.518487648252Mbps', 'rate 6.167734732737Mbps', 'rate 3.328890828601Mbps', 'rate 7.562283742982Mbps', 'rate 3.602763597442Mbps', 'rate 3.523095638174Mbps', 'rate 3.031498667220Mbps', 'rate 7.708643676837Mbps', 'rate 7.993020010556Mbps', 'rate 11.889148702790Mbps', 'rate 10.458698493694Mbps', 'rate 6.650944541242Mbps', 'rate 11.236465503077Mbps', 'rate 8.875177619953Mbps', 'rate 9.879370638496Mbps', 'rate 11.222464168270Mbps', 'rate 3.744741421064Mbps', 'rate 8.090697274489Mbps', 'rate 10.505221764332Mbps', 'rate 10.371647338877Mbps', 'rate 3.484415150475Mbps', 'rate 7.964369635364Mbps', 'rate 6.800441367605Mbps', 'rate 3.670954666344Mbps', 'rate 11.647034382816Mbps', 'rate 3.806283213747Mbps', 'rate 9.930038022273Mbps', 'rate 11.133182916147Mbps', 'rate 6.302926085976Mbps', 'rate 3.825343483626Mbps', 'rate 9.996045036764Mbps', 'rate 7.160098140339Mbps', 'rate 11.867979104536Mbps', 'rate 11.143047577338Mbps', 'rate 11.384863293039Mbps', 'rate 8.166623625948Mbps', 'rate 7.238636737571Mbps', 'rate 11.086712568373Mbps', 'rate 3.172775385411Mbps', 'rate 3.947050045981Mbps', 'rate 10.389605399813Mbps', 'rate 11.815968301092Mbps', 'rate 11.789555271543Mbps', 'rate 3.449654833847Mbps', 'rate 10.057470201151Mbps', 'rate 11.347087397105Mbps', 'rate 3.586660990528Mbps', 'rate 10.495833268243Mbps', 'rate 3.892782384479Mbps', 'rate 6.739786563097Mbps', 'rate 7.279636438893Mbps', 'rate 10.666043738254Mbps', 'rate 3.153866296049Mbps', 'rate 9.792008346838Mbps', 'rate 10.769275696434Mbps', 'rate 10.238424522599Mbps', 'rate 3.988571724010Mbps', 'rate 10.054914042944Mbps', 'rate 8.707096254643Mbps', 'rate 7.290535340543Mbps', 'rate 10.253867102381Mbps', 'rate 10.859926692251Mbps', 'rate 10.295632500803Mbps', 'rate 11.347709535635Mbps', 'rate 11.301383995485Mbps', 'rate 9.546866765475Mbps', 'rate 11.592512442633Mbps', 'rate 10.122890291870Mbps', 'rate 11.609320448705Mbps', 'rate 3.383002096972Mbps', 'rate 3.130174673184Mbps', 'rate 3.926454996233Mbps', 'rate 11.228980752668Mbps', 'rate 3.123905013230Mbps', 'rate 10.646581391424Mbps', 'rate 7.845064542950Mbps', 'rate 7.145785581382Mbps', 'rate 6.045739947892Mbps', 'rate 3.724168861631Mbps', 'rate 10.029260496138Mbps', 'rate 3.549520905373Mbps', 'rate 10.796563010573Mbps', 'rate 8.510484506158Mbps', 'rate 11.061862792372Mbps', 'rate 11.828572530775Mbps', 'rate 10.335067346798Mbps', 'rate 3.531696163198Mbps', 'rate 11.224213403358Mbps', 'rate 11.164558212016Mbps', 'rate 3.125878029834Mbps', 'rate 7.790292307980Mbps', 'rate 8.858789329796Mbps', 'rate 11.490272633598Mbps', 'rate 8.433141706963Mbps', 'rate 10.269337685916Mbps', 'rate 11.624549045356Mbps', 'rate 11.061375196983Mbps', 'rate 7.159444897421Mbps', 'rate 7.367620270678Mbps', 'rate 3.352656134328Mbps', 'rate 9.360452723024Mbps', 'rate 10.429823491151Mbps', 'rate 3.797289309875Mbps', 'rate 3.735584524274Mbps', 'rate 6.642539033097Mbps', 'rate 3.012201500514Mbps', 'rate 10.923998277911Mbps', 'rate 8.649523941422Mbps', 'rate 9.445591782132Mbps', 'rate 3.063295818549Mbps', 'rate 11.854722229802Mbps', 'rate 3.668384327148Mbps', 'rate 9.840529825057Mbps', 'rate 10.439003849722Mbps', 'rate 10.896543290575Mbps', 'rate 3.010154294646Mbps', 'rate 3.072663105617Mbps', 'rate 10.639500167379Mbps', 'rate 10.285059455540Mbps', 'rate 7.720640298315Mbps', 'rate 11.301686374397Mbps', 'rate 7.298048940737Mbps', 'rate 8.617078604747Mbps', 'rate 11.120908695313Mbps', 'rate 10.257359908920Mbps', 'rate 7.374320149639Mbps', 'rate 6.537907786949Mbps', 'rate 6.015217663014Mbps', 'rate 10.028758724179Mbps', 'rate 6.743905264466Mbps', 'rate 6.302073439724Mbps', 'rate 8.580244534077Mbps', 'rate 3.394808223365Mbps', 'rate 11.731559190048Mbps', 'rate 3.333410985428Mbps', 'rate 10.636944547325Mbps', 'rate 10.865548069706Mbps', 'rate 7.815339622078Mbps', 'rate 7.182271419850Mbps', 'rate 9.846485947747Mbps', 'rate 10.940959036678Mbps', 'rate 6.385145888144Mbps', 'rate 8.417646694946Mbps', 'rate 10.697102200394Mbps', 'rate 3.663110050234Mbps', 'rate 3.050241896577Mbps', 'rate 10.398309160800Mbps', 'rate 10.889061392625Mbps', 'rate 11.742348586312Mbps', 'rate 10.774234838508Mbps', 'rate 3.432545289959Mbps', 'rate 10.084603011158Mbps', 'rate 10.563159487585Mbps', 'rate 10.480572713751Mbps', 'rate 3.610833579130Mbps', 'rate 10.833647512440Mbps', 'rate 9.297490368246Mbps', 'rate 3.685971724408Mbps', 'rate 10.531510423926Mbps', 'rate 8.551742069703Mbps', 'rate 6.397812770573Mbps', 'rate 11.262826314298Mbps', 'rate 6.087401978994Mbps', 'rate 3.614383240350Mbps', 'rate 3.141850517560Mbps', 'rate 10.163774945610Mbps', 'rate 10.452219517576Mbps', 'rate 10.173180948439Mbps', 'rate 3.334322895765Mbps', 'rate 7.788861733774Mbps', 'rate 10.013365036755Mbps', 'rate 11.716199944103Mbps', 'rate 3.031681712245Mbps', 'rate 11.722711808215Mbps', 'rate 10.906437546269Mbps', 'rate 3.177440872206Mbps', 'rate 6.322267660306Mbps', 'rate 11.875273567703Mbps', 'rate 3.191743039011Mbps', 'rate 10.518116282916Mbps', 'rate 11.453621536460Mbps', 'rate 8.372240024791Mbps', 'rate 6.825464990244Mbps', 'rate 3.622729977311Mbps', 'rate 10.527687091323Mbps', 'rate 3.416819212603Mbps', 'rate 3.363973448894Mbps', 'rate 11.894756326335Mbps', 'rate 9.021674387124Mbps', 'rate 3.038072652937Mbps', 'rate 3.654980798382Mbps', 'rate 10.350827091685Mbps', 'rate 10.913157537846Mbps', 'rate 3.051923484928Mbps', 'rate 3.974354760075Mbps', 'rate 3.044922165827Mbps', 'rate 10.476374580144Mbps', 'rate 9.536317248862Mbps', 'rate 6.721083031362Mbps', 'rate 8.933127900990Mbps', 'rate 3.463616896137Mbps', 'rate 11.850080150835Mbps', 'rate 3.898623646998Mbps', 'rate 3.714608353275Mbps', 'rate 10.286261572930Mbps', 'rate 3.763761088086Mbps', 'rate 8.457813772735Mbps', 'rate 11.258168177565Mbps', 'rate 3.707333685600Mbps', 'rate 3.735920326352Mbps', 'rate 7.086811044129Mbps', 'rate 3.383357935407Mbps', 'rate 7.467640114075Mbps', 'rate 11.130516556867Mbps', 'rate 3.753260404315Mbps', 'rate 11.760005937585Mbps', 'rate 3.309240219922Mbps', 'rate 10.523226528164Mbps', 'rate 7.284058993980Mbps', 'rate 11.526980262541Mbps', 'rate 6.353014523819Mbps', 'rate 11.127729304638Mbps', 'rate 8.650046494480Mbps', 'rate 7.654554951436Mbps', 'rate 8.099660427572Mbps', 'rate 9.515533704214Mbps', 'rate 10.605074056626Mbps', 'rate 11.126228508510Mbps', 'rate 3.891820888245Mbps', 'rate 8.449557675321Mbps', 'rate 3.898735577114Mbps', 'rate 11.380801538843Mbps', 'rate 10.524397403436Mbps', 'rate 3.704112399658Mbps', 'rate 10.354982951125Mbps', 'rate 11.123502228066Mbps', 'rate 11.764119224872Mbps', 'rate 10.810431451061Mbps', 'rate 6.093196876056Mbps', 'rate 9.911504008257Mbps', 'rate 3.643378253773Mbps', 'rate 9.269921521668Mbps', 'rate 7.856001442784Mbps', 'rate 7.710440933074Mbps', 'rate 11.268907304608Mbps', 'rate 3.106445898448Mbps', 'rate 10.510295529378Mbps', 'rate 10.562360948002Mbps', 'rate 3.544378214098Mbps', 'rate 3.178914794088Mbps', 'rate 10.322366164965Mbps', 'rate 3.950182639867Mbps', 'rate 6.499849561799Mbps', 'rate 6.127563305268Mbps', 'rate 3.547853966020Mbps', 'rate 10.027873119656Mbps', 'rate 9.381513392649Mbps', 'rate 10.707865733729Mbps', 'rate 3.736099029961Mbps', 'rate 3.078466065861Mbps', 'rate 7.149772382553Mbps', 'rate 10.035255200336Mbps', 'rate 10.159206594002Mbps', 'rate 9.551674324759Mbps', 'rate 3.687044310056Mbps', 'rate 11.239663435715Mbps', 'rate 7.331600613219Mbps', 'rate 6.435426533312Mbps', 'rate 3.129384518365Mbps', 'rate 8.950710506274Mbps', 'rate 3.058634151548Mbps', 'rate 7.872463242323Mbps', 'rate 3.167103381526Mbps', 'rate 3.610169824134Mbps', 'rate 10.527750929373Mbps', 'rate 3.082342361945Mbps', 'rate 11.875737195147Mbps', 'rate 10.632027704965Mbps', 'rate 10.684577867422Mbps', 'rate 10.125367339147Mbps', 'rate 8.419088497572Mbps', 'rate 10.660147349429Mbps', 'rate 10.177304099659Mbps', 'rate 10.481911698074Mbps', 'rate 11.170139927155Mbps', 'rate 7.644611190330Mbps', 'rate 10.877298673188Mbps', 'rate 9.135243652695Mbps', 'rate 9.996424391587Mbps', 'rate 11.422668507479Mbps', 'rate 10.521873360258Mbps', 'rate 3.792036323801Mbps', 'rate 10.007072708908Mbps', 'rate 6.694526077474Mbps', 'rate 10.154063175295Mbps', 'rate 9.940148884512Mbps', 'rate 3.077519375486Mbps', 'rate 11.179169241252Mbps', 'rate 11.871771446066Mbps', 'rate 3.649733229357Mbps', 'rate 11.604771798791Mbps', 'rate 9.873985367994Mbps', 'rate 11.157746036348Mbps', 'rate 9.884360120858Mbps', 'rate 3.705577697803Mbps', 'rate 3.319708519443Mbps', 'rate 7.316753495860Mbps', 'rate 11.642622676892Mbps', 'rate 10.955522851721Mbps', 'rate 10.707983502067Mbps', 'rate 3.754626767336Mbps', 'rate 3.178635920105Mbps', 'rate 11.020759351142Mbps', 'rate 8.348418299617Mbps', 'rate 9.545498614925Mbps', 'rate 8.749958468725Mbps', 'rate 10.460990636290Mbps', 'rate 3.329374484124Mbps', 'rate 10.457811132793Mbps', 'rate 7.467192175335Mbps', 'rate 11.326667004251Mbps', 'rate 6.406409850653Mbps', 'rate 11.843893875859Mbps', 'rate 10.941531932680Mbps', 'rate 3.690600552146Mbps', 'rate 9.577965512903Mbps', 'rate 6.939266839505Mbps', 'rate 3.520986833613Mbps', 'rate 7.867148810665Mbps', 'rate 10.472149337529Mbps', 'rate 10.230820652924Mbps', 'rate 10.532175899295Mbps', 'rate 9.919914047454Mbps', 'rate 3.381289958569Mbps', 'rate 11.201949182299Mbps', 'rate 3.201941940887Mbps', 'rate 3.684267398086Mbps', 'rate 11.221920060758Mbps', 'rate 11.896174433092Mbps', 'rate 3.059884783664Mbps', 'rate 10.110532615474Mbps', 'rate 3.104803796505Mbps', 'rate 7.347397501454Mbps', 'rate 11.691789945969Mbps', 'rate 10.514071332266Mbps', 'rate 3.133522657509Mbps', 'rate 7.066663927205Mbps', 'rate 8.970500614754Mbps', 'rate 11.134236777104Mbps', 'rate 10.349040302163Mbps', 'rate 3.792076236491Mbps', 'rate 10.395079609120Mbps', 'rate 3.253015779838Mbps', 'rate 10.234388433866Mbps', 'rate 10.263425214364Mbps', 'rate 10.482558872701Mbps', 'rate 10.738452466677Mbps', 'rate 3.158527042427Mbps', 'rate 3.744826185875Mbps', 'rate 3.691618472038Mbps', 'rate 10.681320821427Mbps', 'rate 7.518627328922Mbps', 'rate 8.733307297992Mbps', 'rate 3.951459042108Mbps', 'rate 9.899484839727Mbps', 'rate 8.726264017591Mbps', 'rate 6.117693430738Mbps', 'rate 9.874578004112Mbps', 'rate 11.101603318703Mbps', 'rate 3.013883713476Mbps', 'rate 11.719890390974Mbps', 'rate 3.495230324114Mbps', 'rate 6.050230191228Mbps', 'rate 7.426016732540Mbps', 'rate 10.864155170098Mbps', 'rate 3.795008974785Mbps', 'rate 6.849173531601Mbps', 'rate 11.802224696759Mbps', 'rate 11.606445649158Mbps', 'rate 10.864933873482Mbps', 'rate 3.549077885373Mbps', 'rate 11.788368769383Mbps', 'rate 10.078395951019Mbps', 'rate 9.929501496814Mbps', 'rate 3.482530338014Mbps', 'rate 6.920509680697Mbps', 'rate 11.119101851679Mbps', 'rate 3.476250828778Mbps', 'rate 10.857048248678Mbps', 'rate 11.502655690522Mbps', 'rate 3.863740644093Mbps', 'rate 3.852500844578Mbps', 'rate 9.428424274016Mbps', 'rate 3.358787136214Mbps', 'rate 3.958741494003Mbps', 'rate 3.073292298872Mbps', 'rate 11.782288960863Mbps', 'rate 10.375057804902Mbps', 'rate 7.040784049764Mbps', 'rate 10.435475262920Mbps', 'rate 3.015035164077Mbps', 'rate 10.937874102685Mbps', 'rate 3.222109739771Mbps', 'rate 11.779444044119Mbps', 'rate 3.570265217716Mbps', 'rate 8.712480236944Mbps', 'rate 11.545064631309Mbps', 'rate 10.824715062332Mbps', 'rate 6.323268102105Mbps', 'rate 6.012005256039Mbps', 'rate 10.018672251462Mbps', 'rate 11.545065476525Mbps', 'rate 3.756822305474Mbps', 'rate 10.950907200752Mbps', 'rate 3.537594428866Mbps', 'rate 9.226484417366Mbps', 'rate 3.579574814841Mbps', 'rate 6.234665477198Mbps', 'rate 3.420918054403Mbps', 'rate 7.763209837594Mbps', 'rate 3.309557478517Mbps', 'rate 11.121444173047Mbps', 'rate 6.772817817042Mbps', 'rate 10.931264716565Mbps', 'rate 9.056702834472Mbps', 'rate 11.510186476470Mbps', 'rate 11.524873636026Mbps', 'rate 3.537126851851Mbps', 'rate 11.277309213397Mbps', 'rate 3.618682391318Mbps', 'rate 9.905508443070Mbps', 'rate 3.679707922305Mbps', 'rate 10.525860197315Mbps', 'rate 11.467291651117Mbps', 'rate 3.065079026196Mbps', 'rate 3.596041152231Mbps', 'rate 7.799717163743Mbps', 'rate 3.470404015421Mbps', 'rate 9.305785150396Mbps', 'rate 3.934196347026Mbps', 'rate 7.437664099522Mbps', 'rate 11.148548323843Mbps', 'rate 3.039611934988Mbps', 'rate 3.951206945276Mbps', 'rate 3.497100090845Mbps', 'rate 6.790525634575Mbps', 'rate 3.663946417916Mbps', 'rate 10.077954939862Mbps', 'rate 3.339863057405Mbps', 'rate 3.138769705761Mbps', 'rate 10.381191672492Mbps', 'rate 3.793221136042Mbps', 'rate 3.590081685838Mbps', 'rate 3.850372281331Mbps', 'rate 3.455394126662Mbps', 'rate 6.507102117369Mbps', 'rate 10.367457233603Mbps', 'rate 6.817002280795Mbps', 'rate 8.050092574813Mbps', 'rate 10.954339731294Mbps', 'rate 3.417125966251Mbps', 'rate 6.675456324058Mbps', 'rate 3.549570672820Mbps', 'rate 3.152201832558Mbps', 'rate 10.612805251076Mbps', 'rate 3.726173438420Mbps', 'rate 6.090569306196Mbps', 'rate 3.955441121557Mbps', 'rate 7.605905038866Mbps', 'rate 3.897974925468Mbps', 'rate 10.378954248657Mbps', 'rate 11.602798118514Mbps', 'rate 10.127138491099Mbps', 'rate 11.069762470683Mbps', 'rate 10.021226761843Mbps', 'rate 3.299344776381Mbps', 'rate 9.348117282414Mbps', 'rate 3.227159192628Mbps', 'rate 3.150789076820Mbps', 'rate 10.974426231854Mbps', 'rate 3.339719771188Mbps', 'rate 7.502424449819Mbps', 'rate 11.548753596893Mbps', 'rate 6.572777103179Mbps', 'rate 3.217814277372Mbps', 'rate 7.177908126957Mbps', 'rate 3.304807969889Mbps', 'rate 7.580442451505Mbps', 'rate 3.415102025797Mbps', 'rate 11.481521735929Mbps', 'rate 11.418824357530Mbps', 'rate 10.234145410440Mbps', 'rate 3.089201018380Mbps', 'rate 3.565402258752Mbps', 'rate 11.874790831728Mbps', 'rate 9.177213991698Mbps', 'rate 11.785310886621Mbps', 'rate 11.056758955752Mbps', 'rate 10.348864579135Mbps', 'rate 9.351772007950Mbps', 'rate 9.973053654066Mbps', 'rate 3.848025122163Mbps', 'rate 3.325515899824Mbps', 'rate 6.565697226559Mbps', 'rate 6.203900318892Mbps', 'rate 10.492630335431Mbps', 'rate 3.040099591958Mbps', 'rate 7.794822396444Mbps', 'rate 6.456373868121Mbps', 'rate 6.976125991579Mbps', 'rate 7.263792796760Mbps', 'rate 10.155311279445Mbps', 'rate 8.840937188800Mbps', 'rate 3.396617703309Mbps', 'rate 8.299633323381Mbps', 'rate 11.663725753454Mbps', 'rate 10.947527996093Mbps', 'rate 8.897639345722Mbps', 'rate 9.743713794855Mbps', 'rate 3.425530874186Mbps', 'rate 11.743557992420Mbps', 'rate 6.432855052344Mbps', 'rate 3.324731471323Mbps', 'rate 10.590423277497Mbps', 'rate 8.985298162564Mbps', 'rate 8.141532568472Mbps', 'rate 9.866090517493Mbps', 'rate 3.474894080155Mbps', 'rate 3.812064256661Mbps', 'rate 3.520068995770Mbps', 'rate 10.068255668474Mbps', 'rate 11.239150958895Mbps', 'rate 11.583371306598Mbps', 'rate 11.226396225234Mbps', 'rate 3.825324672710Mbps', 'rate 3.922057850570Mbps', 'rate 7.231608739893Mbps', 'rate 10.079561075243Mbps', 'rate 3.640927626200Mbps', 'rate 11.218241624890Mbps', 'rate 7.536321308753Mbps', 'rate 10.401672913901Mbps', 'rate 10.501961774454Mbps', 'rate 9.191992445201Mbps', 'rate 7.047772982982Mbps', 'rate 10.295043547935Mbps', 'rate 10.822764315395Mbps', 'rate 10.611134644413Mbps', 'rate 10.515259116480Mbps', 'rate 11.770214595791Mbps', 'rate 8.959682214548Mbps', 'rate 3.910432995744Mbps', 'rate 3.773498599754Mbps', 'rate 3.085757165581Mbps', 'rate 3.590901533155Mbps', 'rate 9.087005691563Mbps', 'rate 3.027065128062Mbps', 'rate 3.893705156140Mbps', 'rate 9.265937597747Mbps', 'rate 11.015261146246Mbps', 'rate 3.206607228400Mbps', 'rate 10.722281481708Mbps', 'rate 10.664657192281Mbps', 'rate 7.482122211072Mbps', 'rate 7.994304986806Mbps', 'rate 3.760822570899Mbps', 'rate 3.913928624036Mbps', 'rate 8.610758941483Mbps', 'rate 3.973228211569Mbps', 'rate 3.697516202101Mbps', 'rate 3.151797430868Mbps', 'rate 10.388728938346Mbps', 'rate 10.097372851849Mbps', 'rate 10.370767877989Mbps', 'rate 3.358756363312Mbps', 'rate 11.342296400596Mbps', 'rate 6.344916565184Mbps', 'rate 10.448205793793Mbps', 'rate 6.290790095021Mbps', 'rate 11.471024691386Mbps', 'rate 3.000850766278Mbps', 'rate 7.466107031980Mbps', 'rate 3.322624878726Mbps', 'rate 11.392324422885Mbps', 'rate 7.158355873584Mbps', 'rate 10.842593543675Mbps', 'rate 10.118901777449Mbps', 'rate 8.768945187185Mbps', 'rate 11.168419959229Mbps', 'rate 11.006179467075Mbps', 'rate 3.246947616082Mbps', 'rate 3.427814129658Mbps', 'rate 10.430783281161Mbps', 'rate 3.795810890066Mbps', 'rate 7.441104518560Mbps', 'rate 3.731844267089Mbps', 'rate 10.692411616590Mbps', 'rate 6.003028596917Mbps', 'rate 8.282758102486Mbps', 'rate 10.268739844233Mbps', 'rate 6.471787883235Mbps', 'rate 8.045928637416Mbps', 'rate 3.873310149607Mbps', 'rate 7.495372911080Mbps', 'rate 11.266724985156Mbps', 'rate 10.958678601544Mbps', 'rate 8.065694573442Mbps', 'rate 6.740008872724Mbps', 'rate 10.438115708446Mbps', 'rate 11.079095987655Mbps', 'rate 8.697620857815Mbps', 'rate 7.336057575893Mbps', 'rate 11.838944979492Mbps', 'rate 6.855986874243Mbps', 'rate 6.451479362417Mbps', 'rate 6.096212537547Mbps', 'rate 3.579319728599Mbps', 'rate 8.972444995091Mbps', 'rate 10.225048348685Mbps', 'rate 11.242538335406Mbps', 'rate 11.096453039361Mbps', 'rate 3.802982319717Mbps', 'rate 3.006092829404Mbps', 'rate 3.278838452648Mbps', 'rate 6.125653893540Mbps', 'rate 3.498199963003Mbps', 'rate 3.196870666110Mbps', 'rate 3.485982049025Mbps', 'rate 11.177909063722Mbps', 'rate 7.955099886342Mbps', 'rate 11.779135140581Mbps', 'rate 11.433842195680Mbps', 'rate 10.262215958252Mbps', 'rate 7.858587248927Mbps', 'rate 11.404853230745Mbps', 'rate 3.801690585688Mbps', 'rate 9.420240584848Mbps', 'rate 9.783550714606Mbps', 'rate 10.009219812304Mbps', 'rate 11.866335465046Mbps', 'rate 10.937930782251Mbps', 'rate 8.521726232465Mbps', 'rate 10.913587305195Mbps', 'rate 3.777701757267Mbps', 'rate 3.765866288959Mbps', 'rate 3.495554975227Mbps', 'rate 10.325799818666Mbps', 'rate 3.511850422915Mbps', 'rate 10.714937657406Mbps', 'rate 3.449496782060Mbps', 'rate 3.226178362597Mbps', 'rate 3.982677370210Mbps', 'rate 9.309987516368Mbps', 'rate 10.087845995387Mbps', 'rate 3.099555764700Mbps', 'rate 10.942441217576Mbps', 'rate 7.883725085027Mbps', 'rate 11.194855608533Mbps', 'rate 6.042404692938Mbps', 'rate 11.176054136865Mbps', 'rate 11.242678476722Mbps', 'rate 9.308401937253Mbps', 'rate 3.862948594540Mbps', 'rate 11.569957170875Mbps', 'rate 11.428316381598Mbps', 'rate 7.340122129001Mbps', 'rate 3.012278512409Mbps', 'rate 7.925157611840Mbps', 'rate 6.615336620292Mbps', 'rate 3.744956589933Mbps', 'rate 7.607266732004Mbps', 'rate 3.173082648005Mbps', 'rate 11.069956763096Mbps', 'rate 10.151243741086Mbps', 'rate 11.840474146531Mbps', 'rate 6.678762706648Mbps', 'rate 3.559996797111Mbps', 'rate 11.573743030033Mbps', 'rate 10.585629340463Mbps', 'rate 8.803010649833Mbps', 'rate 11.430191382986Mbps', 'rate 7.945615939096Mbps', 'rate 10.232212448923Mbps', 'rate 8.575831781484Mbps', 'rate 3.791153644550Mbps', 'rate 3.006395852123Mbps', 'rate 10.878672491870Mbps', 'rate 11.112768919513Mbps', 'rate 3.537302723271Mbps', 'rate 8.483300790363Mbps', 'rate 3.684539993149Mbps', 'rate 3.544196704429Mbps', 'rate 7.699348672148Mbps', 'rate 6.100305552515Mbps', 'rate 3.112766320674Mbps', 'rate 3.089624371612Mbps', 'rate 7.015080363477Mbps', 'rate 9.239474849017Mbps', 'rate 7.027472477480Mbps', 'rate 10.454217230562Mbps', 'rate 3.172211036240Mbps', 'rate 10.544719809625Mbps', 'rate 8.473360812594Mbps', 'rate 3.876634708541Mbps', 'rate 10.561515041787Mbps', 'rate 7.447962006138Mbps', 'rate 3.036564902322Mbps', 'rate 10.446267007408Mbps', 'rate 11.790787625653Mbps', 'rate 10.772227888737Mbps', 'rate 8.570791881815Mbps', 'rate 8.336013352637Mbps', 'rate 8.657365472717Mbps', 'rate 3.720436523093Mbps', 'rate 3.365882166813Mbps', 'rate 3.398281238681Mbps', 'rate 3.913693203003Mbps', 'rate 6.149052777962Mbps', 'rate 6.018796328272Mbps', 'rate 3.393441067085Mbps', 'rate 11.855676904433Mbps', 'rate 3.358870395468Mbps', 'rate 10.115958667993Mbps', 'rate 10.809266474882Mbps', 'rate 3.048283427176Mbps', 'rate 11.266037235701Mbps', 'rate 7.656017899844Mbps', 'rate 3.475680252551Mbps', 'rate 9.661556749362Mbps', 'rate 3.776033876788Mbps', 'rate 3.669021013650Mbps', 'rate 11.350965022893Mbps', 'rate 10.042056658942Mbps', 'rate 9.834942117135Mbps', 'rate 3.149268402355Mbps', 'rate 9.018729052705Mbps', 'rate 3.533923027009Mbps', 'rate 8.563318384068Mbps', 'rate 3.150195331651Mbps', 'rate 7.641122388760Mbps', 'rate 11.779191116216Mbps', 'rate 6.642995292249Mbps', 'rate 3.911181062025Mbps', 'rate 3.498645677569Mbps', 'rate 3.083523476842Mbps', 'rate 10.185894071834Mbps', 'rate 3.743674799571Mbps', 'rate 7.254324540745Mbps', 'rate 10.106344191517Mbps', 'rate 3.304422406782Mbps', 'rate 7.213766378050Mbps', 'rate 10.559677874335Mbps', 'rate 10.479252823723Mbps', 'rate 8.457251916894Mbps', 'rate 3.590282184778Mbps', 'rate 10.790762895816Mbps', 'rate 3.715556365340Mbps', 'rate 6.368792689628Mbps', 'rate 11.502604522915Mbps', 'rate 10.920423832302Mbps', 'rate 3.801706661629Mbps', 'rate 11.200837762079Mbps', 'rate 7.627987007544Mbps', 'rate 7.496213024101Mbps', 'rate 3.483763124761Mbps', 'rate 8.604060108265Mbps', 'rate 3.783274174865Mbps', 'rate 7.406359916327Mbps', 'rate 6.948240913714Mbps', 'rate 3.092183863977Mbps', 'rate 11.771878028188Mbps', 'rate 9.478024829700Mbps', 'rate 3.683510248285Mbps', 'rate 11.791942463414Mbps', 'rate 10.580879976307Mbps', 'rate 10.707234396793Mbps', 'rate 8.496002339270Mbps', 'rate 11.770346841060Mbps', 'rate 11.573947570241Mbps', 'rate 6.415053595343Mbps', 'rate 8.322971762834Mbps', 'rate 11.161616869293Mbps', 'rate 10.842123284766Mbps', 'rate 11.334320215352Mbps', 'rate 3.158367293582Mbps', 'rate 8.564152196388Mbps', 'rate 8.456341082986Mbps', 'rate 3.591586799854Mbps', 'rate 11.497691357122Mbps', 'rate 3.805684759677Mbps', 'rate 6.111709493097Mbps', 'rate 11.674899394575Mbps', 'rate 11.361497653577Mbps', 'rate 6.479491102910Mbps', 'rate 11.387587015382Mbps', 'rate 11.402670857796Mbps', 'rate 8.601510459242Mbps', 'rate 9.522247833691Mbps', 'rate 9.444820402109Mbps', 'rate 11.258670906950Mbps', 'rate 11.848386396846Mbps', 'rate 11.496675434493Mbps', 'rate 7.700299676791Mbps', 'rate 6.898653143125Mbps', 'rate 9.235825671644Mbps', 'rate 6.481564436466Mbps', 'rate 10.331971434176Mbps', 'rate 3.986662677187Mbps', 'rate 11.666350355680Mbps', 'rate 3.943880030745Mbps', 'rate 3.530362219523Mbps', 'rate 3.725494408063Mbps', 'rate 3.529531475697Mbps', 'rate 11.835854231761Mbps', 'rate 3.925770053720Mbps', 'rate 10.582522726079Mbps', 'rate 10.317339191642Mbps', 'rate 3.187601855313Mbps', 'rate 3.578444881221Mbps', 'rate 3.304504013248Mbps', 'rate 11.373305632584Mbps', 'rate 10.501466174271Mbps', 'rate 3.337131478240Mbps', 'rate 3.613471250311Mbps', 'rate 6.696349456359Mbps', 'rate 8.326888745334Mbps', 'rate 3.258079834409Mbps', 'rate 11.700179583009Mbps', 'rate 3.115438227674Mbps', 'rate 11.735623807014Mbps', 'rate 6.373330240091Mbps', 'rate 10.424969882028Mbps', 'rate 9.184504474646Mbps', 'rate 6.677308636281Mbps', 'rate 10.849045595806Mbps', 'rate 6.987864875093Mbps', 'rate 6.952720890959Mbps', 'rate 10.684455507903Mbps', 'rate 3.636828197398Mbps', 'rate 6.053764184093Mbps', 'rate 11.052308047815Mbps', 'rate 8.808512742884Mbps', 'rate 3.899629780261Mbps', 'rate 10.702493534803Mbps', 'rate 6.424235641640Mbps', 'rate 11.491744774936Mbps', 'rate 9.177002957332Mbps', 'rate 6.682933848514Mbps', 'rate 10.948404757952Mbps', 'rate 8.978279291260Mbps', 'rate 3.120141205751Mbps', 'rate 11.405446625925Mbps', 'rate 3.414112724338Mbps', 'rate 3.839233008398Mbps', 'rate 9.009903978128Mbps', 'rate 3.821211547413Mbps', 'rate 3.316032295405Mbps', 'rate 3.453309689083Mbps', 'rate 3.408526188193Mbps', 'rate 6.574527848507Mbps', 'rate 11.656533304824Mbps', 'rate 8.227460471071Mbps', 'rate 3.927046000453Mbps', 'rate 3.463023580485Mbps', 'rate 10.450525497134Mbps', 'rate 10.518960789443Mbps', 'rate 11.031558876581Mbps', 'rate 10.875128444427Mbps', 'rate 10.160323539610Mbps', 'rate 11.023724784980Mbps', 'rate 3.780859515613Mbps', 'rate 8.809589968598Mbps', 'rate 10.785875735077Mbps', 'rate 3.536317253939Mbps', 'rate 10.654663495637Mbps', 'rate 3.068760113454Mbps', 'rate 11.340372483253Mbps', 'rate 8.178432490075Mbps', 'rate 11.274990585314Mbps', 'rate 3.217410177488Mbps', 'rate 8.141994262826Mbps', 'rate 10.217570026129Mbps', 'rate 3.627218070526Mbps', 'rate 11.602468358051Mbps', 'rate 11.189693904020Mbps', 'rate 3.865255553691Mbps', 'rate 8.546992813986Mbps', 'rate 10.027344315568Mbps', 'rate 10.691947451322Mbps', 'rate 3.532648657678Mbps', 'rate 11.566701161740Mbps', 'rate 10.675351608141Mbps', 'rate 7.318403530380Mbps', 'rate 11.761531237907Mbps', 'rate 11.068409571240Mbps', 'rate 7.829661613442Mbps', 'rate 10.956873720784Mbps', 'rate 11.016514114657Mbps', 'rate 10.042823948617Mbps', 'rate 10.360275175643Mbps', 'rate 10.517874993950Mbps', 'rate 6.791433893411Mbps', 'rate 3.649279023073Mbps', 'rate 11.549294595172Mbps', 'rate 6.878148657681Mbps', 'rate 7.767698935742Mbps', 'rate 3.976431724794Mbps', 'rate 11.319219764476Mbps', 'rate 3.000498750536Mbps', 'rate 3.996646977448Mbps', 'rate 3.141895769796Mbps', 'rate 9.844021279834Mbps', 'rate 3.158796246516Mbps', 'rate 10.443015491238Mbps', 'rate 3.900767336874Mbps', 'rate 10.481114192937Mbps', 'rate 10.374563210161Mbps', 'rate 3.249220156698Mbps', 'rate 11.330682329098Mbps']

        
        conn = SinkConnector(
            pipeline_config=app_cfg,
            agent=agent,
            mqtt_config=mqtt_cfg,
            network_controller=network_controller,
            feed_name=feed_name
        )

    await conn.webrtc_coro()

def update_loop(num_workers, result_queue, update_queue):
    """
    Main loop to average weights and distribute them among workers.
    """
    while True:
        weights = []
        for _ in range(num_workers):
            try:
                weight = result_queue.get(timeout=9999)
                result_queue.task_done()
                weights.append(weight)
            except Empty:
                LOGGER.info("Timeout waiting for weight update from agents")

        averaged_weights = average_weights(weights)
        for _ in range(num_workers):
            try:
                update_queue.put(averaged_weights, timeout=9999)
            except Full:
                LOGGER.info("Timeout waiting for update queue to be free")
        update_queue.join()

if __name__ == "__main__":
    with Manager() as manager:
        result_queue = manager.JoinableQueue(maxsize=NUM_WORKERS)
        update_queue = manager.JoinableQueue(maxsize=NUM_WORKERS)
        workers = create_workers(NUM_WORKERS, result_queue, update_queue, WEIGHT_AGGREGATION_FREQUENCY)
        update_loop(NUM_WORKERS, result_queue, update_queue)
