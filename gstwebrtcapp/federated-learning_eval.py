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
        network_controller.rules = ['rate 10.295984749727Mbps', 'rate 10.893509829463Mbps', 'rate 10.249737799605Mbps', 'rate 9.866684073629Mbps', 'rate 8.330054987626Mbps', 'rate 3.454909283673Mbps', 'rate 10.167162173637Mbps', 'rate 10.339174095166Mbps', 'rate 9.813910955653Mbps', 'rate 10.402768242888Mbps', 'rate 11.358967476581Mbps', 'rate 3.152151451236Mbps', 'rate 2.897560482896Mbps', 'rate 10.367875691681Mbps', 'rate 2.962503489973Mbps', 'rate 6.325626994975Mbps', 'rate 11.030672988981Mbps', 'rate 10.957762597168Mbps', 'rate 2.549078521788Mbps', 'rate 10.239065774439Mbps', 'rate 3.389903666907Mbps', 'rate 8.867391080155Mbps', 'rate 3.324306080930Mbps', 'rate 2.692685797840Mbps', 'rate 2.686041716005Mbps', 'rate 9.903196262721Mbps', 'rate 3.203561894298Mbps', 'rate 2.802443442075Mbps', 'rate 9.701513707428Mbps', 'rate 2.910814029226Mbps', 'rate 10.058720176662Mbps', 'rate 2.521790831755Mbps', 'rate 2.577485365821Mbps', 'rate 3.471277373948Mbps', 'rate 10.735678956229Mbps', 'rate 10.482674951671Mbps', 'rate 11.287526849219Mbps', 'rate 9.007232370024Mbps', 'rate 2.901523256595Mbps', 'rate 10.154496096526Mbps', 'rate 2.840767698688Mbps', 'rate 3.128600390916Mbps', 'rate 11.184440600925Mbps', 'rate 11.057353285958Mbps', 'rate 9.754508721843Mbps', 'rate 9.554247242836Mbps', 'rate 6.332785209820Mbps', 'rate 3.072438930118Mbps', 'rate 3.326550951148Mbps', 'rate 2.939976290823Mbps', 'rate 3.416218127180Mbps', 'rate 7.891753168314Mbps', 'rate 10.383194046604Mbps', 'rate 2.702675475322Mbps', 'rate 9.793378041323Mbps', 'rate 10.881967540470Mbps', 'rate 11.267159116774Mbps', 'rate 9.598097858454Mbps', 'rate 8.159052258404Mbps', 'rate 6.225635145466Mbps', 'rate 6.754192568661Mbps', 'rate 9.565587353074Mbps', 'rate 11.203997866125Mbps', 'rate 8.625193226739Mbps', 'rate 5.892147723330Mbps', 'rate 3.424592523093Mbps', 'rate 10.001935980495Mbps', 'rate 3.249493367220Mbps', 'rate 3.188084212776Mbps', 'rate 8.148800539511Mbps', 'rate 5.619569963433Mbps', 'rate 9.955362317100Mbps', 'rate 2.930098552524Mbps', 'rate 2.578501710999Mbps', 'rate 8.208403239637Mbps', 'rate 2.688809870844Mbps', 'rate 6.443017781430Mbps', 'rate 11.327810347715Mbps', 'rate 9.288337357613Mbps', 'rate 10.124014099682Mbps', 'rate 9.963318483249Mbps', 'rate 7.018487648252Mbps', 'rate 5.667734732737Mbps', 'rate 2.828890828601Mbps', 'rate 7.062283742982Mbps', 'rate 3.102763597442Mbps', 'rate 3.023095638174Mbps', 'rate 2.531498667220Mbps', 'rate 7.208643676837Mbps', 'rate 7.493020010556Mbps', 'rate 11.389148702790Mbps', 'rate 9.958698493694Mbps', 'rate 6.150944541242Mbps', 'rate 10.736465503077Mbps', 'rate 8.375177619953Mbps', 'rate 9.379370638496Mbps', 'rate 10.722464168270Mbps', 'rate 3.244741421064Mbps', 'rate 7.590697274489Mbps', 'rate 10.005221764332Mbps', 'rate 9.871647338877Mbps', 'rate 2.984415150475Mbps', 'rate 7.464369635364Mbps', 'rate 6.300441367605Mbps', 'rate 3.170954666344Mbps', 'rate 11.147034382816Mbps', 'rate 3.306283213747Mbps', 'rate 9.430038022273Mbps', 'rate 10.633182916147Mbps', 'rate 5.802926085976Mbps', 'rate 3.325343483626Mbps', 'rate 9.496045036764Mbps', 'rate 6.660098140339Mbps', 'rate 11.367979104536Mbps', 'rate 10.643047577338Mbps', 'rate 10.884863293039Mbps', 'rate 7.666623625948Mbps', 'rate 6.738636737571Mbps', 'rate 10.586712568373Mbps', 'rate 2.672775385411Mbps', 'rate 3.447050045981Mbps', 'rate 9.889605399813Mbps', 'rate 11.315968301092Mbps', 'rate 11.289555271543Mbps', 'rate 2.949654833847Mbps', 'rate 9.557470201151Mbps', 'rate 10.847087397105Mbps', 'rate 3.086660990528Mbps', 'rate 9.995833268243Mbps', 'rate 3.392782384479Mbps', 'rate 6.239786563097Mbps', 'rate 6.779636438893Mbps', 'rate 10.166043738254Mbps', 'rate 2.653866296049Mbps', 'rate 9.292008346838Mbps', 'rate 10.269275696434Mbps', 'rate 9.738424522599Mbps', 'rate 3.488571724010Mbps', 'rate 9.554914042944Mbps', 'rate 8.207096254643Mbps', 'rate 6.790535340543Mbps', 'rate 9.753867102381Mbps', 'rate 10.359926692251Mbps', 'rate 9.795632500803Mbps', 'rate 10.847709535635Mbps', 'rate 10.801383995485Mbps', 'rate 9.046866765475Mbps', 'rate 11.092512442633Mbps', 'rate 9.622890291870Mbps', 'rate 11.109320448705Mbps', 'rate 2.883002096972Mbps', 'rate 2.630174673184Mbps', 'rate 3.426454996233Mbps', 'rate 10.728980752668Mbps', 'rate 2.623905013230Mbps', 'rate 10.146581391424Mbps', 'rate 7.345064542950Mbps', 'rate 6.645785581382Mbps', 'rate 5.545739947892Mbps', 'rate 3.224168861631Mbps', 'rate 9.529260496138Mbps', 'rate 3.049520905373Mbps', 'rate 10.296563010573Mbps', 'rate 8.010484506158Mbps', 'rate 10.561862792372Mbps', 'rate 11.328572530775Mbps', 'rate 9.835067346798Mbps', 'rate 3.031696163198Mbps', 'rate 10.724213403358Mbps', 'rate 10.664558212016Mbps', 'rate 2.625878029834Mbps', 'rate 7.290292307980Mbps', 'rate 8.358789329796Mbps', 'rate 10.990272633598Mbps', 'rate 7.933141706963Mbps', 'rate 9.769337685916Mbps', 'rate 11.124549045356Mbps', 'rate 10.561375196983Mbps', 'rate 6.659444897421Mbps', 'rate 6.867620270678Mbps', 'rate 2.852656134328Mbps', 'rate 8.860452723024Mbps', 'rate 9.929823491151Mbps', 'rate 3.297289309875Mbps', 'rate 3.235584524274Mbps', 'rate 6.142539033097Mbps', 'rate 2.512201500514Mbps', 'rate 10.423998277911Mbps', 'rate 8.149523941422Mbps', 'rate 8.945591782132Mbps', 'rate 2.563295818549Mbps', 'rate 11.354722229802Mbps', 'rate 3.168384327148Mbps', 'rate 9.340529825057Mbps', 'rate 9.939003849722Mbps', 'rate 10.396543290575Mbps', 'rate 2.510154294646Mbps', 'rate 2.572663105617Mbps', 'rate 10.139500167379Mbps', 'rate 9.785059455540Mbps', 'rate 7.220640298315Mbps', 'rate 10.801686374397Mbps', 'rate 6.798048940737Mbps', 'rate 8.117078604747Mbps', 'rate 10.620908695313Mbps', 'rate 9.757359908920Mbps', 'rate 6.874320149639Mbps', 'rate 6.037907786949Mbps', 'rate 5.515217663014Mbps', 'rate 9.528758724179Mbps', 'rate 6.243905264466Mbps', 'rate 5.802073439724Mbps', 'rate 8.080244534077Mbps', 'rate 2.894808223365Mbps', 'rate 11.231559190048Mbps', 'rate 2.833410985428Mbps', 'rate 10.136944547325Mbps', 'rate 10.365548069706Mbps', 'rate 7.315339622078Mbps', 'rate 6.682271419850Mbps', 'rate 9.346485947747Mbps', 'rate 10.440959036678Mbps', 'rate 5.885145888144Mbps', 'rate 7.917646694946Mbps', 'rate 10.197102200394Mbps', 'rate 3.163110050234Mbps', 'rate 2.550241896577Mbps', 'rate 9.898309160800Mbps', 'rate 10.389061392625Mbps', 'rate 11.242348586312Mbps', 'rate 10.274234838508Mbps', 'rate 2.932545289959Mbps', 'rate 9.584603011158Mbps', 'rate 10.063159487585Mbps', 'rate 9.980572713751Mbps', 'rate 3.110833579130Mbps', 'rate 10.333647512440Mbps', 'rate 8.797490368246Mbps', 'rate 3.185971724408Mbps', 'rate 10.031510423926Mbps', 'rate 8.051742069703Mbps', 'rate 5.897812770573Mbps', 'rate 10.762826314298Mbps', 'rate 5.587401978994Mbps', 'rate 3.114383240350Mbps', 'rate 2.641850517560Mbps', 'rate 9.663774945610Mbps', 'rate 9.952219517576Mbps', 'rate 9.673180948439Mbps', 'rate 2.834322895765Mbps', 'rate 7.288861733774Mbps', 'rate 9.513365036755Mbps', 'rate 11.216199944103Mbps', 'rate 2.531681712245Mbps', 'rate 11.222711808215Mbps', 'rate 10.406437546269Mbps', 'rate 2.677440872206Mbps', 'rate 5.822267660306Mbps', 'rate 11.375273567703Mbps', 'rate 2.691743039011Mbps', 'rate 10.018116282916Mbps', 'rate 10.953621536460Mbps', 'rate 7.872240024791Mbps', 'rate 6.325464990244Mbps', 'rate 3.122729977311Mbps', 'rate 10.027687091323Mbps', 'rate 2.916819212603Mbps', 'rate 2.863973448894Mbps', 'rate 11.394756326335Mbps', 'rate 8.521674387124Mbps', 'rate 2.538072652937Mbps', 'rate 3.154980798382Mbps', 'rate 9.850827091685Mbps', 'rate 10.413157537846Mbps', 'rate 2.551923484928Mbps', 'rate 3.474354760075Mbps', 'rate 2.544922165827Mbps', 'rate 9.976374580144Mbps', 'rate 9.036317248862Mbps', 'rate 6.221083031362Mbps', 'rate 8.433127900990Mbps', 'rate 2.963616896137Mbps', 'rate 11.350080150835Mbps', 'rate 3.398623646998Mbps', 'rate 3.214608353275Mbps', 'rate 9.786261572930Mbps', 'rate 3.263761088086Mbps', 'rate 7.957813772735Mbps', 'rate 10.758168177565Mbps', 'rate 3.207333685600Mbps', 'rate 3.235920326352Mbps', 'rate 6.586811044129Mbps', 'rate 2.883357935407Mbps', 'rate 6.967640114075Mbps', 'rate 10.630516556867Mbps', 'rate 3.253260404315Mbps', 'rate 11.260005937585Mbps', 'rate 2.809240219922Mbps', 'rate 10.023226528164Mbps', 'rate 6.784058993980Mbps', 'rate 11.026980262541Mbps', 'rate 5.853014523819Mbps', 'rate 10.627729304638Mbps', 'rate 8.150046494480Mbps', 'rate 7.154554951436Mbps', 'rate 7.599660427572Mbps', 'rate 9.015533704214Mbps', 'rate 10.105074056626Mbps', 'rate 10.626228508510Mbps', 'rate 3.391820888245Mbps', 'rate 7.949557675321Mbps', 'rate 3.398735577114Mbps', 'rate 10.880801538843Mbps', 'rate 10.024397403436Mbps', 'rate 3.204112399658Mbps', 'rate 9.854982951125Mbps', 'rate 10.623502228066Mbps', 'rate 11.264119224872Mbps', 'rate 10.310431451061Mbps', 'rate 5.593196876056Mbps', 'rate 9.411504008257Mbps', 'rate 3.143378253773Mbps', 'rate 8.769921521668Mbps', 'rate 7.356001442784Mbps', 'rate 7.210440933074Mbps', 'rate 10.768907304608Mbps', 'rate 2.606445898448Mbps', 'rate 10.010295529378Mbps', 'rate 10.062360948002Mbps', 'rate 3.044378214098Mbps', 'rate 2.678914794088Mbps', 'rate 9.822366164965Mbps', 'rate 3.450182639867Mbps', 'rate 5.999849561799Mbps', 'rate 5.627563305268Mbps', 'rate 3.047853966020Mbps', 'rate 9.527873119656Mbps', 'rate 8.881513392649Mbps', 'rate 10.207865733729Mbps', 'rate 3.236099029961Mbps', 'rate 2.578466065861Mbps', 'rate 6.649772382553Mbps', 'rate 9.535255200336Mbps', 'rate 9.659206594002Mbps', 'rate 9.051674324759Mbps', 'rate 3.187044310056Mbps', 'rate 10.739663435715Mbps', 'rate 6.831600613219Mbps', 'rate 5.935426533312Mbps', 'rate 2.629384518365Mbps', 'rate 8.450710506274Mbps', 'rate 2.558634151548Mbps', 'rate 7.372463242323Mbps', 'rate 2.667103381526Mbps', 'rate 3.110169824134Mbps', 'rate 10.027750929373Mbps', 'rate 2.582342361945Mbps', 'rate 11.375737195147Mbps', 'rate 10.132027704965Mbps', 'rate 10.184577867422Mbps', 'rate 9.625367339147Mbps', 'rate 7.919088497572Mbps', 'rate 10.160147349429Mbps', 'rate 9.677304099659Mbps', 'rate 9.981911698074Mbps', 'rate 10.670139927155Mbps', 'rate 7.144611190330Mbps', 'rate 10.377298673188Mbps', 'rate 8.635243652695Mbps', 'rate 9.496424391587Mbps', 'rate 10.922668507479Mbps', 'rate 10.021873360258Mbps', 'rate 3.292036323801Mbps', 'rate 9.507072708908Mbps', 'rate 6.194526077474Mbps', 'rate 9.654063175295Mbps', 'rate 9.440148884512Mbps', 'rate 2.577519375486Mbps', 'rate 10.679169241252Mbps', 'rate 11.371771446066Mbps', 'rate 3.149733229357Mbps', 'rate 11.104771798791Mbps', 'rate 9.373985367994Mbps', 'rate 10.657746036348Mbps', 'rate 9.384360120858Mbps', 'rate 3.205577697803Mbps', 'rate 2.819708519443Mbps', 'rate 6.816753495860Mbps', 'rate 11.142622676892Mbps', 'rate 10.455522851721Mbps', 'rate 10.207983502067Mbps', 'rate 3.254626767336Mbps', 'rate 2.678635920105Mbps', 'rate 10.520759351142Mbps', 'rate 7.848418299617Mbps', 'rate 9.045498614925Mbps', 'rate 8.249958468725Mbps', 'rate 9.960990636290Mbps', 'rate 2.829374484124Mbps', 'rate 9.957811132793Mbps', 'rate 6.967192175335Mbps', 'rate 10.826667004251Mbps', 'rate 5.906409850653Mbps', 'rate 11.343893875859Mbps', 'rate 10.441531932680Mbps', 'rate 3.190600552146Mbps', 'rate 9.077965512903Mbps', 'rate 6.439266839505Mbps', 'rate 3.020986833613Mbps', 'rate 7.367148810665Mbps', 'rate 9.972149337529Mbps', 'rate 9.730820652924Mbps', 'rate 10.032175899295Mbps', 'rate 9.419914047454Mbps', 'rate 2.881289958569Mbps', 'rate 10.701949182299Mbps', 'rate 2.701941940887Mbps', 'rate 3.184267398086Mbps', 'rate 10.721920060758Mbps', 'rate 11.396174433092Mbps', 'rate 2.559884783664Mbps', 'rate 9.610532615474Mbps', 'rate 2.604803796505Mbps', 'rate 6.847397501454Mbps', 'rate 11.191789945969Mbps', 'rate 10.014071332266Mbps', 'rate 2.633522657509Mbps', 'rate 6.566663927205Mbps', 'rate 8.470500614754Mbps', 'rate 10.634236777104Mbps', 'rate 9.849040302163Mbps', 'rate 3.292076236491Mbps', 'rate 9.895079609120Mbps', 'rate 2.753015779838Mbps', 'rate 9.734388433866Mbps', 'rate 9.763425214364Mbps', 'rate 9.982558872701Mbps', 'rate 10.238452466677Mbps', 'rate 2.658527042427Mbps', 'rate 3.244826185875Mbps', 'rate 3.191618472038Mbps', 'rate 10.181320821427Mbps', 'rate 7.018627328922Mbps', 'rate 8.233307297992Mbps', 'rate 3.451459042108Mbps', 'rate 9.399484839727Mbps', 'rate 8.226264017591Mbps', 'rate 5.617693430738Mbps', 'rate 9.374578004112Mbps', 'rate 10.601603318703Mbps', 'rate 2.513883713476Mbps', 'rate 11.219890390974Mbps', 'rate 2.995230324114Mbps', 'rate 5.550230191228Mbps', 'rate 6.926016732540Mbps', 'rate 10.364155170098Mbps', 'rate 3.295008974785Mbps', 'rate 6.349173531601Mbps', 'rate 11.302224696759Mbps', 'rate 11.106445649158Mbps', 'rate 10.364933873482Mbps', 'rate 3.049077885373Mbps', 'rate 11.288368769383Mbps', 'rate 9.578395951019Mbps', 'rate 9.429501496814Mbps', 'rate 2.982530338014Mbps', 'rate 6.420509680697Mbps', 'rate 10.619101851679Mbps', 'rate 2.976250828778Mbps', 'rate 10.357048248678Mbps', 'rate 11.002655690522Mbps', 'rate 3.363740644093Mbps', 'rate 3.352500844578Mbps', 'rate 8.928424274016Mbps', 'rate 2.858787136214Mbps', 'rate 3.458741494003Mbps', 'rate 2.573292298872Mbps', 'rate 11.282288960863Mbps', 'rate 9.875057804902Mbps', 'rate 6.540784049764Mbps', 'rate 9.935475262920Mbps', 'rate 2.515035164077Mbps', 'rate 10.437874102685Mbps', 'rate 2.722109739771Mbps', 'rate 11.279444044119Mbps', 'rate 3.070265217716Mbps', 'rate 8.212480236944Mbps', 'rate 11.045064631309Mbps', 'rate 10.324715062332Mbps', 'rate 5.823268102105Mbps', 'rate 5.512005256039Mbps', 'rate 9.518672251462Mbps', 'rate 11.045065476525Mbps', 'rate 3.256822305474Mbps', 'rate 10.450907200752Mbps', 'rate 3.037594428866Mbps', 'rate 8.726484417366Mbps', 'rate 3.079574814841Mbps', 'rate 5.734665477198Mbps', 'rate 2.920918054403Mbps', 'rate 7.263209837594Mbps', 'rate 2.809557478517Mbps', 'rate 10.621444173047Mbps', 'rate 6.272817817042Mbps', 'rate 10.431264716565Mbps', 'rate 8.556702834472Mbps', 'rate 11.010186476470Mbps', 'rate 11.024873636026Mbps', 'rate 3.037126851851Mbps', 'rate 10.777309213397Mbps', 'rate 3.118682391318Mbps', 'rate 9.405508443070Mbps', 'rate 3.179707922305Mbps', 'rate 10.025860197315Mbps', 'rate 10.967291651117Mbps', 'rate 2.565079026196Mbps', 'rate 3.096041152231Mbps', 'rate 7.299717163743Mbps', 'rate 2.970404015421Mbps', 'rate 8.805785150396Mbps', 'rate 3.434196347026Mbps', 'rate 6.937664099522Mbps', 'rate 10.648548323843Mbps', 'rate 2.539611934988Mbps', 'rate 3.451206945276Mbps', 'rate 2.997100090845Mbps', 'rate 6.290525634575Mbps', 'rate 3.163946417916Mbps', 'rate 9.577954939862Mbps', 'rate 2.839863057405Mbps', 'rate 2.638769705761Mbps', 'rate 9.881191672492Mbps', 'rate 3.293221136042Mbps', 'rate 3.090081685838Mbps', 'rate 3.350372281331Mbps', 'rate 2.955394126662Mbps', 'rate 6.007102117369Mbps', 'rate 9.867457233603Mbps', 'rate 6.317002280795Mbps', 'rate 7.550092574813Mbps', 'rate 10.454339731294Mbps', 'rate 2.917125966251Mbps', 'rate 6.175456324058Mbps', 'rate 3.049570672820Mbps', 'rate 2.652201832558Mbps', 'rate 10.112805251076Mbps', 'rate 3.226173438420Mbps', 'rate 5.590569306196Mbps', 'rate 3.455441121557Mbps', 'rate 7.105905038866Mbps', 'rate 3.397974925468Mbps', 'rate 9.878954248657Mbps', 'rate 11.102798118514Mbps', 'rate 9.627138491099Mbps', 'rate 10.569762470683Mbps', 'rate 9.521226761843Mbps', 'rate 2.799344776381Mbps', 'rate 8.848117282414Mbps', 'rate 2.727159192628Mbps', 'rate 2.650789076820Mbps', 'rate 10.474426231854Mbps', 'rate 2.839719771188Mbps', 'rate 7.002424449819Mbps', 'rate 11.048753596893Mbps', 'rate 6.072777103179Mbps', 'rate 2.717814277372Mbps', 'rate 6.677908126957Mbps', 'rate 2.804807969889Mbps', 'rate 7.080442451505Mbps', 'rate 2.915102025797Mbps', 'rate 10.981521735929Mbps', 'rate 10.918824357530Mbps', 'rate 9.734145410440Mbps', 'rate 2.589201018380Mbps', 'rate 3.065402258752Mbps', 'rate 11.374790831728Mbps', 'rate 8.677213991698Mbps', 'rate 11.285310886621Mbps', 'rate 10.556758955752Mbps', 'rate 9.848864579135Mbps', 'rate 8.851772007950Mbps', 'rate 9.473053654066Mbps', 'rate 3.348025122163Mbps', 'rate 2.825515899824Mbps', 'rate 6.065697226559Mbps', 'rate 5.703900318892Mbps', 'rate 9.992630335431Mbps', 'rate 2.540099591958Mbps', 'rate 7.294822396444Mbps', 'rate 5.956373868121Mbps', 'rate 6.476125991579Mbps', 'rate 6.763792796760Mbps', 'rate 9.655311279445Mbps', 'rate 8.340937188800Mbps', 'rate 2.896617703309Mbps', 'rate 7.799633323381Mbps', 'rate 11.163725753454Mbps', 'rate 10.447527996093Mbps', 'rate 8.397639345722Mbps', 'rate 9.243713794855Mbps', 'rate 2.925530874186Mbps', 'rate 11.243557992420Mbps', 'rate 5.932855052344Mbps', 'rate 2.824731471323Mbps', 'rate 10.090423277497Mbps', 'rate 8.485298162564Mbps', 'rate 7.641532568472Mbps', 'rate 9.366090517493Mbps', 'rate 2.974894080155Mbps', 'rate 3.312064256661Mbps', 'rate 3.020068995770Mbps', 'rate 9.568255668474Mbps', 'rate 10.739150958895Mbps', 'rate 11.083371306598Mbps', 'rate 10.726396225234Mbps', 'rate 3.325324672710Mbps', 'rate 3.422057850570Mbps', 'rate 6.731608739893Mbps', 'rate 9.579561075243Mbps', 'rate 3.140927626200Mbps', 'rate 10.718241624890Mbps', 'rate 7.036321308753Mbps', 'rate 9.901672913901Mbps', 'rate 10.001961774454Mbps', 'rate 8.691992445201Mbps', 'rate 6.547772982982Mbps', 'rate 9.795043547935Mbps', 'rate 10.322764315395Mbps', 'rate 10.111134644413Mbps', 'rate 10.015259116480Mbps', 'rate 11.270214595791Mbps', 'rate 8.459682214548Mbps', 'rate 3.410432995744Mbps', 'rate 3.273498599754Mbps', 'rate 2.585757165581Mbps', 'rate 3.090901533155Mbps', 'rate 8.587005691563Mbps', 'rate 2.527065128062Mbps', 'rate 3.393705156140Mbps', 'rate 8.765937597747Mbps', 'rate 10.515261146246Mbps', 'rate 2.706607228400Mbps', 'rate 10.222281481708Mbps', 'rate 10.164657192281Mbps', 'rate 6.982122211072Mbps', 'rate 7.494304986806Mbps', 'rate 3.260822570899Mbps', 'rate 3.413928624036Mbps', 'rate 8.110758941483Mbps', 'rate 3.473228211569Mbps', 'rate 3.197516202101Mbps', 'rate 2.651797430868Mbps', 'rate 9.888728938346Mbps', 'rate 9.597372851849Mbps', 'rate 9.870767877989Mbps', 'rate 2.858756363312Mbps', 'rate 10.842296400596Mbps', 'rate 5.844916565184Mbps', 'rate 9.948205793793Mbps', 'rate 5.790790095021Mbps', 'rate 10.971024691386Mbps', 'rate 2.500850766278Mbps', 'rate 6.966107031980Mbps', 'rate 2.822624878726Mbps', 'rate 10.892324422885Mbps', 'rate 6.658355873584Mbps', 'rate 10.342593543675Mbps', 'rate 9.618901777449Mbps', 'rate 8.268945187185Mbps', 'rate 10.668419959229Mbps', 'rate 10.506179467075Mbps', 'rate 2.746947616082Mbps', 'rate 2.927814129658Mbps', 'rate 9.930783281161Mbps', 'rate 3.295810890066Mbps', 'rate 6.941104518560Mbps', 'rate 3.231844267089Mbps', 'rate 10.192411616590Mbps', 'rate 5.503028596917Mbps', 'rate 7.782758102486Mbps', 'rate 9.768739844233Mbps', 'rate 5.971787883235Mbps', 'rate 7.545928637416Mbps', 'rate 3.373310149607Mbps', 'rate 6.995372911080Mbps', 'rate 10.766724985156Mbps', 'rate 10.458678601544Mbps', 'rate 7.565694573442Mbps', 'rate 6.240008872724Mbps', 'rate 9.938115708446Mbps', 'rate 10.579095987655Mbps', 'rate 8.197620857815Mbps', 'rate 6.836057575893Mbps', 'rate 11.338944979492Mbps', 'rate 6.355986874243Mbps', 'rate 5.951479362417Mbps', 'rate 5.596212537547Mbps', 'rate 3.079319728599Mbps', 'rate 8.472444995091Mbps', 'rate 9.725048348685Mbps', 'rate 10.742538335406Mbps', 'rate 10.596453039361Mbps', 'rate 3.302982319717Mbps', 'rate 2.506092829404Mbps', 'rate 2.778838452648Mbps', 'rate 5.625653893540Mbps', 'rate 2.998199963003Mbps', 'rate 2.696870666110Mbps', 'rate 2.985982049025Mbps', 'rate 10.677909063722Mbps', 'rate 7.455099886342Mbps', 'rate 11.279135140581Mbps', 'rate 10.933842195680Mbps', 'rate 9.762215958252Mbps', 'rate 7.358587248927Mbps', 'rate 10.904853230745Mbps', 'rate 3.301690585688Mbps', 'rate 8.920240584848Mbps', 'rate 9.283550714606Mbps', 'rate 9.509219812304Mbps', 'rate 11.366335465046Mbps', 'rate 10.437930782251Mbps', 'rate 8.021726232465Mbps', 'rate 10.413587305195Mbps', 'rate 3.277701757267Mbps', 'rate 3.265866288959Mbps', 'rate 2.995554975227Mbps', 'rate 9.825799818666Mbps', 'rate 3.011850422915Mbps', 'rate 10.214937657406Mbps', 'rate 2.949496782060Mbps', 'rate 2.726178362597Mbps', 'rate 3.482677370210Mbps', 'rate 8.809987516368Mbps', 'rate 9.587845995387Mbps', 'rate 2.599555764700Mbps', 'rate 10.442441217576Mbps', 'rate 7.383725085027Mbps', 'rate 10.694855608533Mbps', 'rate 5.542404692938Mbps', 'rate 10.676054136865Mbps', 'rate 10.742678476722Mbps', 'rate 8.808401937253Mbps', 'rate 3.362948594540Mbps', 'rate 11.069957170875Mbps', 'rate 10.928316381598Mbps', 'rate 6.840122129001Mbps', 'rate 2.512278512409Mbps', 'rate 7.425157611840Mbps', 'rate 6.115336620292Mbps', 'rate 3.244956589933Mbps', 'rate 7.107266732004Mbps', 'rate 2.673082648005Mbps', 'rate 10.569956763096Mbps', 'rate 9.651243741086Mbps', 'rate 11.340474146531Mbps', 'rate 6.178762706648Mbps', 'rate 3.059996797111Mbps', 'rate 11.073743030033Mbps', 'rate 10.085629340463Mbps', 'rate 8.303010649833Mbps', 'rate 10.930191382986Mbps', 'rate 7.445615939096Mbps', 'rate 9.732212448923Mbps', 'rate 8.075831781484Mbps', 'rate 3.291153644550Mbps', 'rate 2.506395852123Mbps', 'rate 10.378672491870Mbps', 'rate 10.612768919513Mbps', 'rate 3.037302723271Mbps', 'rate 7.983300790363Mbps', 'rate 3.184539993149Mbps', 'rate 3.044196704429Mbps', 'rate 7.199348672148Mbps', 'rate 5.600305552515Mbps', 'rate 2.612766320674Mbps', 'rate 2.589624371612Mbps', 'rate 6.515080363477Mbps', 'rate 8.739474849017Mbps', 'rate 6.527472477480Mbps', 'rate 9.954217230562Mbps', 'rate 2.672211036240Mbps', 'rate 10.044719809625Mbps', 'rate 7.973360812594Mbps', 'rate 3.376634708541Mbps', 'rate 10.061515041787Mbps', 'rate 6.947962006138Mbps', 'rate 2.536564902322Mbps', 'rate 9.946267007408Mbps', 'rate 11.290787625653Mbps', 'rate 10.272227888737Mbps', 'rate 8.070791881815Mbps', 'rate 7.836013352637Mbps', 'rate 8.157365472717Mbps', 'rate 3.220436523093Mbps', 'rate 2.865882166813Mbps', 'rate 2.898281238681Mbps', 'rate 3.413693203003Mbps', 'rate 5.649052777962Mbps', 'rate 5.518796328272Mbps', 'rate 2.893441067085Mbps', 'rate 11.355676904433Mbps', 'rate 2.858870395468Mbps', 'rate 9.615958667993Mbps', 'rate 10.309266474882Mbps', 'rate 2.548283427176Mbps', 'rate 10.766037235701Mbps', 'rate 7.156017899844Mbps', 'rate 2.975680252551Mbps', 'rate 9.161556749362Mbps', 'rate 3.276033876788Mbps', 'rate 3.169021013650Mbps', 'rate 10.850965022893Mbps', 'rate 9.542056658942Mbps', 'rate 9.334942117135Mbps', 'rate 2.649268402355Mbps', 'rate 8.518729052705Mbps', 'rate 3.033923027009Mbps', 'rate 8.063318384068Mbps', 'rate 2.650195331651Mbps', 'rate 7.141122388760Mbps', 'rate 11.279191116216Mbps', 'rate 6.142995292249Mbps', 'rate 3.411181062025Mbps', 'rate 2.998645677569Mbps', 'rate 2.583523476842Mbps', 'rate 9.685894071834Mbps', 'rate 3.243674799571Mbps', 'rate 6.754324540745Mbps', 'rate 9.606344191517Mbps', 'rate 2.804422406782Mbps', 'rate 6.713766378050Mbps', 'rate 10.059677874335Mbps', 'rate 9.979252823723Mbps', 'rate 7.957251916894Mbps', 'rate 3.090282184778Mbps', 'rate 10.290762895816Mbps', 'rate 3.215556365340Mbps', 'rate 5.868792689628Mbps', 'rate 11.002604522915Mbps', 'rate 10.420423832302Mbps', 'rate 3.301706661629Mbps', 'rate 10.700837762079Mbps', 'rate 7.127987007544Mbps', 'rate 6.996213024101Mbps', 'rate 2.983763124761Mbps', 'rate 8.104060108265Mbps', 'rate 3.283274174865Mbps', 'rate 6.906359916327Mbps', 'rate 6.448240913714Mbps', 'rate 2.592183863977Mbps', 'rate 11.271878028188Mbps', 'rate 8.978024829700Mbps', 'rate 3.183510248285Mbps', 'rate 11.291942463414Mbps', 'rate 10.080879976307Mbps', 'rate 10.207234396793Mbps', 'rate 7.996002339270Mbps', 'rate 11.270346841060Mbps', 'rate 11.073947570241Mbps', 'rate 5.915053595343Mbps', 'rate 7.822971762834Mbps', 'rate 10.661616869293Mbps', 'rate 10.342123284766Mbps', 'rate 10.834320215352Mbps', 'rate 2.658367293582Mbps', 'rate 8.064152196388Mbps', 'rate 7.956341082986Mbps', 'rate 3.091586799854Mbps', 'rate 10.997691357122Mbps', 'rate 3.305684759677Mbps', 'rate 5.611709493097Mbps', 'rate 11.174899394575Mbps', 'rate 10.861497653577Mbps', 'rate 5.979491102910Mbps', 'rate 10.887587015382Mbps', 'rate 10.902670857796Mbps', 'rate 8.101510459242Mbps', 'rate 9.022247833691Mbps', 'rate 8.944820402109Mbps', 'rate 10.758670906950Mbps', 'rate 11.348386396846Mbps', 'rate 10.996675434493Mbps', 'rate 7.200299676791Mbps', 'rate 6.398653143125Mbps', 'rate 8.735825671644Mbps', 'rate 5.981564436466Mbps', 'rate 9.831971434176Mbps', 'rate 3.486662677187Mbps', 'rate 11.166350355680Mbps', 'rate 3.443880030745Mbps', 'rate 3.030362219523Mbps', 'rate 3.225494408063Mbps', 'rate 3.029531475697Mbps', 'rate 11.335854231761Mbps', 'rate 3.425770053720Mbps', 'rate 10.082522726079Mbps', 'rate 9.817339191642Mbps', 'rate 2.687601855313Mbps', 'rate 3.078444881221Mbps', 'rate 2.804504013248Mbps', 'rate 10.873305632584Mbps', 'rate 10.001466174271Mbps', 'rate 2.837131478240Mbps', 'rate 3.113471250311Mbps', 'rate 6.196349456359Mbps', 'rate 7.826888745334Mbps', 'rate 2.758079834409Mbps', 'rate 11.200179583009Mbps', 'rate 2.615438227674Mbps', 'rate 11.235623807014Mbps', 'rate 5.873330240091Mbps', 'rate 9.924969882028Mbps', 'rate 8.684504474646Mbps', 'rate 6.177308636281Mbps', 'rate 10.349045595806Mbps', 'rate 6.487864875093Mbps', 'rate 6.452720890959Mbps', 'rate 10.184455507903Mbps', 'rate 3.136828197398Mbps', 'rate 5.553764184093Mbps', 'rate 10.552308047815Mbps', 'rate 8.308512742884Mbps', 'rate 3.399629780261Mbps', 'rate 10.202493534803Mbps', 'rate 5.924235641640Mbps', 'rate 10.991744774936Mbps', 'rate 8.677002957332Mbps', 'rate 6.182933848514Mbps', 'rate 10.448404757952Mbps', 'rate 8.478279291260Mbps', 'rate 2.620141205751Mbps', 'rate 10.905446625925Mbps', 'rate 2.914112724338Mbps', 'rate 3.339233008398Mbps', 'rate 8.509903978128Mbps', 'rate 3.321211547413Mbps', 'rate 2.816032295405Mbps', 'rate 2.953309689083Mbps', 'rate 2.908526188193Mbps', 'rate 6.074527848507Mbps', 'rate 11.156533304824Mbps', 'rate 7.727460471071Mbps', 'rate 3.427046000453Mbps', 'rate 2.963023580485Mbps', 'rate 9.950525497134Mbps', 'rate 10.018960789443Mbps', 'rate 10.531558876581Mbps', 'rate 10.375128444427Mbps', 'rate 9.660323539610Mbps', 'rate 10.523724784980Mbps', 'rate 3.280859515613Mbps', 'rate 8.309589968598Mbps', 'rate 10.285875735077Mbps', 'rate 3.036317253939Mbps', 'rate 10.154663495637Mbps', 'rate 2.568760113454Mbps', 'rate 10.840372483253Mbps', 'rate 7.678432490075Mbps', 'rate 10.774990585314Mbps', 'rate 2.717410177488Mbps', 'rate 7.641994262826Mbps', 'rate 9.717570026129Mbps', 'rate 3.127218070526Mbps', 'rate 11.102468358051Mbps', 'rate 10.689693904020Mbps', 'rate 3.365255553691Mbps', 'rate 8.046992813986Mbps', 'rate 9.527344315568Mbps', 'rate 10.191947451322Mbps', 'rate 3.032648657678Mbps', 'rate 11.066701161740Mbps', 'rate 10.175351608141Mbps', 'rate 6.818403530380Mbps', 'rate 11.261531237907Mbps', 'rate 10.568409571240Mbps', 'rate 7.329661613442Mbps', 'rate 10.456873720784Mbps', 'rate 10.516514114657Mbps', 'rate 9.542823948617Mbps', 'rate 9.860275175643Mbps', 'rate 10.017874993950Mbps', 'rate 6.291433893411Mbps', 'rate 3.149279023073Mbps', 'rate 11.049294595172Mbps', 'rate 6.378148657681Mbps', 'rate 7.267698935742Mbps', 'rate 3.476431724794Mbps', 'rate 10.819219764476Mbps', 'rate 2.500498750536Mbps', 'rate 3.496646977448Mbps', 'rate 2.641895769796Mbps', 'rate 9.344021279834Mbps', 'rate 2.658796246516Mbps', 'rate 9.943015491238Mbps', 'rate 3.400767336874Mbps', 'rate 9.981114192937Mbps', 'rate 9.874563210161Mbps', 'rate 2.749220156698Mbps', 'rate 10.830682329098Mbps']

        
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
