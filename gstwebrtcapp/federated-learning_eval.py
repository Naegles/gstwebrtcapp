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

MODEL_FILE = f"fedModelsReward/fixed_rRate = 0.25 (crash 84)/drl_model_4000_steps.zip"


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
        network_controller.rules = ['rate 9.295984749727Mbps', 'rate 9.893509829463Mbps', 'rate 9.249737799605Mbps', 'rate 8.866684073629Mbps', 'rate 7.330054987626Mbps', 'rate 2.454909283673Mbps', 'rate 9.167162173637Mbps', 'rate 9.339174095166Mbps', 'rate 8.813910955653Mbps', 'rate 9.402768242888Mbps', 'rate 10.358967476581Mbps', 'rate 2.152151451236Mbps', 'rate 1.897560482896Mbps', 'rate 9.367875691681Mbps', 'rate 1.962503489973Mbps', 'rate 5.325626994975Mbps', 'rate 10.030672988981Mbps', 'rate 9.957762597168Mbps', 'rate 1.549078521788Mbps', 'rate 9.239065774439Mbps', 'rate 2.389903666907Mbps', 'rate 7.867391080155Mbps', 'rate 2.324306080930Mbps', 'rate 1.692685797840Mbps', 'rate 1.686041716005Mbps', 'rate 8.903196262721Mbps', 'rate 2.203561894298Mbps', 'rate 1.802443442075Mbps', 'rate 8.701513707428Mbps', 'rate 1.910814029226Mbps', 'rate 9.058720176662Mbps', 'rate 1.521790831755Mbps', 'rate 1.577485365821Mbps', 'rate 2.471277373948Mbps', 'rate 9.735678956229Mbps', 'rate 9.482674951671Mbps', 'rate 10.287526849219Mbps', 'rate 8.007232370024Mbps', 'rate 1.901523256595Mbps', 'rate 9.154496096526Mbps', 'rate 1.840767698688Mbps', 'rate 2.128600390916Mbps', 'rate 10.184440600925Mbps', 'rate 10.057353285958Mbps', 'rate 8.754508721843Mbps', 'rate 8.554247242836Mbps', 'rate 5.332785209820Mbps', 'rate 2.072438930118Mbps', 'rate 2.326550951148Mbps', 'rate 1.939976290823Mbps', 'rate 2.416218127180Mbps', 'rate 6.891753168314Mbps', 'rate 9.383194046604Mbps', 'rate 1.702675475322Mbps', 'rate 8.793378041323Mbps', 'rate 9.881967540470Mbps', 'rate 10.267159116774Mbps', 'rate 8.598097858454Mbps', 'rate 7.159052258404Mbps', 'rate 5.225635145466Mbps', 'rate 5.754192568661Mbps', 'rate 8.565587353074Mbps', 'rate 10.203997866125Mbps', 'rate 7.625193226739Mbps', 'rate 4.892147723330Mbps', 'rate 2.424592523093Mbps', 'rate 9.001935980495Mbps', 'rate 2.249493367220Mbps', 'rate 2.188084212776Mbps', 'rate 7.148800539511Mbps', 'rate 4.619569963433Mbps', 'rate 8.955362317100Mbps', 'rate 1.930098552524Mbps', 'rate 1.578501710999Mbps', 'rate 7.208403239637Mbps', 'rate 1.688809870844Mbps', 'rate 5.443017781430Mbps', 'rate 10.327810347715Mbps', 'rate 8.288337357613Mbps', 'rate 9.124014099682Mbps', 'rate 8.963318483249Mbps', 'rate 6.018487648252Mbps', 'rate 4.667734732737Mbps', 'rate 1.828890828601Mbps', 'rate 6.062283742982Mbps', 'rate 2.102763597442Mbps', 'rate 2.023095638174Mbps', 'rate 1.531498667220Mbps', 'rate 6.208643676837Mbps', 'rate 6.493020010556Mbps', 'rate 10.389148702790Mbps', 'rate 8.958698493694Mbps', 'rate 5.150944541242Mbps', 'rate 9.736465503077Mbps', 'rate 7.375177619953Mbps', 'rate 8.379370638496Mbps', 'rate 9.722464168270Mbps', 'rate 2.244741421064Mbps', 'rate 6.590697274489Mbps', 'rate 9.005221764332Mbps', 'rate 8.871647338877Mbps', 'rate 1.984415150475Mbps', 'rate 6.464369635364Mbps', 'rate 5.300441367605Mbps', 'rate 2.170954666344Mbps', 'rate 10.147034382816Mbps', 'rate 2.306283213747Mbps', 'rate 8.430038022273Mbps', 'rate 9.633182916147Mbps', 'rate 4.802926085976Mbps', 'rate 2.325343483626Mbps', 'rate 8.496045036764Mbps', 'rate 5.660098140339Mbps', 'rate 10.367979104536Mbps', 'rate 9.643047577338Mbps', 'rate 9.884863293039Mbps', 'rate 6.666623625948Mbps', 'rate 5.738636737571Mbps', 'rate 9.586712568373Mbps', 'rate 1.672775385411Mbps', 'rate 2.447050045981Mbps', 'rate 8.889605399813Mbps', 'rate 10.315968301092Mbps', 'rate 10.289555271543Mbps', 'rate 1.949654833847Mbps', 'rate 8.557470201151Mbps', 'rate 9.847087397105Mbps', 'rate 2.086660990528Mbps', 'rate 8.995833268243Mbps', 'rate 2.392782384479Mbps', 'rate 5.239786563097Mbps', 'rate 5.779636438893Mbps', 'rate 9.166043738254Mbps', 'rate 1.653866296049Mbps', 'rate 8.292008346838Mbps', 'rate 9.269275696434Mbps', 'rate 8.738424522599Mbps', 'rate 2.488571724010Mbps', 'rate 8.554914042944Mbps', 'rate 7.207096254643Mbps', 'rate 5.790535340543Mbps', 'rate 8.753867102381Mbps', 'rate 9.359926692251Mbps', 'rate 8.795632500803Mbps', 'rate 9.847709535635Mbps', 'rate 9.801383995485Mbps', 'rate 8.046866765475Mbps', 'rate 10.092512442633Mbps', 'rate 8.622890291870Mbps', 'rate 10.109320448705Mbps', 'rate 1.883002096972Mbps', 'rate 1.630174673184Mbps', 'rate 2.426454996233Mbps', 'rate 9.728980752668Mbps', 'rate 1.623905013230Mbps', 'rate 9.146581391424Mbps', 'rate 6.345064542950Mbps', 'rate 5.645785581382Mbps', 'rate 4.545739947892Mbps', 'rate 2.224168861631Mbps', 'rate 8.529260496138Mbps', 'rate 2.049520905373Mbps', 'rate 9.296563010573Mbps', 'rate 7.010484506158Mbps', 'rate 9.561862792372Mbps', 'rate 10.328572530775Mbps', 'rate 8.835067346798Mbps', 'rate 2.031696163198Mbps', 'rate 9.724213403358Mbps', 'rate 9.664558212016Mbps', 'rate 1.625878029834Mbps', 'rate 6.290292307980Mbps', 'rate 7.358789329796Mbps', 'rate 9.990272633598Mbps', 'rate 6.933141706963Mbps', 'rate 8.769337685916Mbps', 'rate 10.124549045356Mbps', 'rate 9.561375196983Mbps', 'rate 5.659444897421Mbps', 'rate 5.867620270678Mbps', 'rate 1.852656134328Mbps', 'rate 7.860452723024Mbps', 'rate 8.929823491151Mbps', 'rate 2.297289309875Mbps', 'rate 2.235584524274Mbps', 'rate 5.142539033097Mbps', 'rate 1.512201500514Mbps', 'rate 9.423998277911Mbps', 'rate 7.149523941422Mbps', 'rate 7.945591782132Mbps', 'rate 1.563295818549Mbps', 'rate 10.354722229802Mbps', 'rate 2.168384327148Mbps', 'rate 8.340529825057Mbps', 'rate 8.939003849722Mbps', 'rate 9.396543290575Mbps', 'rate 1.510154294646Mbps', 'rate 1.572663105617Mbps', 'rate 9.139500167379Mbps', 'rate 8.785059455540Mbps', 'rate 6.220640298315Mbps', 'rate 9.801686374397Mbps', 'rate 5.798048940737Mbps', 'rate 7.117078604747Mbps', 'rate 9.620908695313Mbps', 'rate 8.757359908920Mbps', 'rate 5.874320149639Mbps', 'rate 5.037907786949Mbps', 'rate 4.515217663014Mbps', 'rate 8.528758724179Mbps', 'rate 5.243905264466Mbps', 'rate 4.802073439724Mbps', 'rate 7.080244534077Mbps', 'rate 1.894808223365Mbps', 'rate 10.231559190048Mbps', 'rate 1.833410985428Mbps', 'rate 9.136944547325Mbps', 'rate 9.365548069706Mbps', 'rate 6.315339622078Mbps', 'rate 5.682271419850Mbps', 'rate 8.346485947747Mbps', 'rate 9.440959036678Mbps', 'rate 4.885145888144Mbps', 'rate 6.917646694946Mbps', 'rate 9.197102200394Mbps', 'rate 2.163110050234Mbps', 'rate 1.550241896577Mbps', 'rate 8.898309160800Mbps', 'rate 9.389061392625Mbps', 'rate 10.242348586312Mbps', 'rate 9.274234838508Mbps', 'rate 1.932545289959Mbps', 'rate 8.584603011158Mbps', 'rate 9.063159487585Mbps', 'rate 8.980572713751Mbps', 'rate 2.110833579130Mbps', 'rate 9.333647512440Mbps', 'rate 7.797490368246Mbps', 'rate 2.185971724408Mbps', 'rate 9.031510423926Mbps', 'rate 7.051742069703Mbps', 'rate 4.897812770573Mbps', 'rate 9.762826314298Mbps', 'rate 4.587401978994Mbps', 'rate 2.114383240350Mbps', 'rate 1.641850517560Mbps', 'rate 8.663774945610Mbps', 'rate 8.952219517576Mbps', 'rate 8.673180948439Mbps', 'rate 1.834322895765Mbps', 'rate 6.288861733774Mbps', 'rate 8.513365036755Mbps', 'rate 10.216199944103Mbps', 'rate 1.531681712245Mbps', 'rate 10.222711808215Mbps', 'rate 9.406437546269Mbps', 'rate 1.677440872206Mbps', 'rate 4.822267660306Mbps', 'rate 10.375273567703Mbps', 'rate 1.691743039011Mbps', 'rate 9.018116282916Mbps', 'rate 9.953621536460Mbps', 'rate 6.872240024791Mbps', 'rate 5.325464990244Mbps', 'rate 2.122729977311Mbps', 'rate 9.027687091323Mbps', 'rate 1.916819212603Mbps', 'rate 1.863973448894Mbps', 'rate 10.394756326335Mbps', 'rate 7.521674387124Mbps', 'rate 1.538072652937Mbps', 'rate 2.154980798382Mbps', 'rate 8.850827091685Mbps', 'rate 9.413157537846Mbps', 'rate 1.551923484928Mbps', 'rate 2.474354760075Mbps', 'rate 1.544922165827Mbps', 'rate 8.976374580144Mbps', 'rate 8.036317248862Mbps', 'rate 5.221083031362Mbps', 'rate 7.433127900990Mbps', 'rate 1.963616896137Mbps', 'rate 10.350080150835Mbps', 'rate 2.398623646998Mbps', 'rate 2.214608353275Mbps', 'rate 8.786261572930Mbps', 'rate 2.263761088086Mbps', 'rate 6.957813772735Mbps', 'rate 9.758168177565Mbps', 'rate 2.207333685600Mbps', 'rate 2.235920326352Mbps', 'rate 5.586811044129Mbps', 'rate 1.883357935407Mbps', 'rate 5.967640114075Mbps', 'rate 9.630516556867Mbps', 'rate 2.253260404315Mbps', 'rate 10.260005937585Mbps', 'rate 1.809240219922Mbps', 'rate 9.023226528164Mbps', 'rate 5.784058993980Mbps', 'rate 10.026980262541Mbps', 'rate 4.853014523819Mbps', 'rate 9.627729304638Mbps', 'rate 7.150046494480Mbps', 'rate 6.154554951436Mbps', 'rate 6.599660427572Mbps', 'rate 8.015533704214Mbps', 'rate 9.105074056626Mbps', 'rate 9.626228508510Mbps', 'rate 2.391820888245Mbps', 'rate 6.949557675321Mbps', 'rate 2.398735577114Mbps', 'rate 9.880801538843Mbps', 'rate 9.024397403436Mbps', 'rate 2.204112399658Mbps', 'rate 8.854982951125Mbps', 'rate 9.623502228066Mbps', 'rate 10.264119224872Mbps', 'rate 9.310431451061Mbps', 'rate 4.593196876056Mbps', 'rate 8.411504008257Mbps', 'rate 2.143378253773Mbps', 'rate 7.769921521668Mbps', 'rate 6.356001442784Mbps', 'rate 6.210440933074Mbps', 'rate 9.768907304608Mbps', 'rate 1.606445898448Mbps', 'rate 9.010295529378Mbps', 'rate 9.062360948002Mbps', 'rate 2.044378214098Mbps', 'rate 1.678914794088Mbps', 'rate 8.822366164965Mbps', 'rate 2.450182639867Mbps', 'rate 4.999849561799Mbps', 'rate 4.627563305268Mbps', 'rate 2.047853966020Mbps', 'rate 8.527873119656Mbps', 'rate 7.881513392649Mbps', 'rate 9.207865733729Mbps', 'rate 2.236099029961Mbps', 'rate 1.578466065861Mbps', 'rate 5.649772382553Mbps', 'rate 8.535255200336Mbps', 'rate 8.659206594002Mbps', 'rate 8.051674324759Mbps', 'rate 2.187044310056Mbps', 'rate 9.739663435715Mbps', 'rate 5.831600613219Mbps', 'rate 4.935426533312Mbps', 'rate 1.629384518365Mbps', 'rate 7.450710506274Mbps', 'rate 1.558634151548Mbps', 'rate 6.372463242323Mbps', 'rate 1.667103381526Mbps', 'rate 2.110169824134Mbps', 'rate 9.027750929373Mbps', 'rate 1.582342361945Mbps', 'rate 10.375737195147Mbps', 'rate 9.132027704965Mbps', 'rate 9.184577867422Mbps', 'rate 8.625367339147Mbps', 'rate 6.919088497572Mbps', 'rate 9.160147349429Mbps', 'rate 8.677304099659Mbps', 'rate 8.981911698074Mbps', 'rate 9.670139927155Mbps', 'rate 6.144611190330Mbps', 'rate 9.377298673188Mbps', 'rate 7.635243652695Mbps', 'rate 8.496424391587Mbps', 'rate 9.922668507479Mbps', 'rate 9.021873360258Mbps', 'rate 2.292036323801Mbps', 'rate 8.507072708908Mbps', 'rate 5.194526077474Mbps', 'rate 8.654063175295Mbps', 'rate 8.440148884512Mbps', 'rate 1.577519375486Mbps', 'rate 9.679169241252Mbps', 'rate 10.371771446066Mbps', 'rate 2.149733229357Mbps', 'rate 10.104771798791Mbps', 'rate 8.373985367994Mbps', 'rate 9.657746036348Mbps', 'rate 8.384360120858Mbps', 'rate 2.205577697803Mbps', 'rate 1.819708519443Mbps', 'rate 5.816753495860Mbps', 'rate 10.142622676892Mbps', 'rate 9.455522851721Mbps', 'rate 9.207983502067Mbps', 'rate 2.254626767336Mbps', 'rate 1.678635920105Mbps', 'rate 9.520759351142Mbps', 'rate 6.848418299617Mbps', 'rate 8.045498614925Mbps', 'rate 7.249958468725Mbps', 'rate 8.960990636290Mbps', 'rate 1.829374484124Mbps', 'rate 8.957811132793Mbps', 'rate 5.967192175335Mbps', 'rate 9.826667004251Mbps', 'rate 4.906409850653Mbps', 'rate 10.343893875859Mbps', 'rate 9.441531932680Mbps', 'rate 2.190600552146Mbps', 'rate 8.077965512903Mbps', 'rate 5.439266839505Mbps', 'rate 2.020986833613Mbps', 'rate 6.367148810665Mbps', 'rate 8.972149337529Mbps', 'rate 8.730820652924Mbps', 'rate 9.032175899295Mbps', 'rate 8.419914047454Mbps', 'rate 1.881289958569Mbps', 'rate 9.701949182299Mbps', 'rate 1.701941940887Mbps', 'rate 2.184267398086Mbps', 'rate 9.721920060758Mbps', 'rate 10.396174433092Mbps', 'rate 1.559884783664Mbps', 'rate 8.610532615474Mbps', 'rate 1.604803796505Mbps', 'rate 5.847397501454Mbps', 'rate 10.191789945969Mbps', 'rate 9.014071332266Mbps', 'rate 1.633522657509Mbps', 'rate 5.566663927205Mbps', 'rate 7.470500614754Mbps', 'rate 9.634236777104Mbps', 'rate 8.849040302163Mbps', 'rate 2.292076236491Mbps', 'rate 8.895079609120Mbps', 'rate 1.753015779838Mbps', 'rate 8.734388433866Mbps', 'rate 8.763425214364Mbps', 'rate 8.982558872701Mbps', 'rate 9.238452466677Mbps', 'rate 1.658527042427Mbps', 'rate 2.244826185875Mbps', 'rate 2.191618472038Mbps', 'rate 9.181320821427Mbps', 'rate 6.018627328922Mbps', 'rate 7.233307297992Mbps', 'rate 2.451459042108Mbps', 'rate 8.399484839727Mbps', 'rate 7.226264017591Mbps', 'rate 4.617693430738Mbps', 'rate 8.374578004112Mbps', 'rate 9.601603318703Mbps', 'rate 1.513883713476Mbps', 'rate 10.219890390974Mbps', 'rate 1.995230324114Mbps', 'rate 4.550230191228Mbps', 'rate 5.926016732540Mbps', 'rate 9.364155170098Mbps', 'rate 2.295008974785Mbps', 'rate 5.349173531601Mbps', 'rate 10.302224696759Mbps', 'rate 10.106445649158Mbps', 'rate 9.364933873482Mbps', 'rate 2.049077885373Mbps', 'rate 10.288368769383Mbps', 'rate 8.578395951019Mbps', 'rate 8.429501496814Mbps', 'rate 1.982530338014Mbps', 'rate 5.420509680697Mbps', 'rate 9.619101851679Mbps', 'rate 1.976250828778Mbps', 'rate 9.357048248678Mbps', 'rate 10.002655690522Mbps', 'rate 2.363740644093Mbps', 'rate 2.352500844578Mbps', 'rate 7.928424274016Mbps', 'rate 1.858787136214Mbps', 'rate 2.458741494003Mbps', 'rate 1.573292298872Mbps', 'rate 10.282288960863Mbps', 'rate 8.875057804902Mbps', 'rate 5.540784049764Mbps', 'rate 8.935475262920Mbps', 'rate 1.515035164077Mbps', 'rate 9.437874102685Mbps', 'rate 1.722109739771Mbps', 'rate 10.279444044119Mbps', 'rate 2.070265217716Mbps', 'rate 7.212480236944Mbps', 'rate 10.045064631309Mbps', 'rate 9.324715062332Mbps', 'rate 4.823268102105Mbps', 'rate 4.512005256039Mbps', 'rate 8.518672251462Mbps', 'rate 10.045065476525Mbps', 'rate 2.256822305474Mbps', 'rate 9.450907200752Mbps', 'rate 2.037594428866Mbps', 'rate 7.726484417366Mbps', 'rate 2.079574814841Mbps', 'rate 4.734665477198Mbps', 'rate 1.920918054403Mbps', 'rate 6.263209837594Mbps', 'rate 1.809557478517Mbps', 'rate 9.621444173047Mbps', 'rate 5.272817817042Mbps', 'rate 9.431264716565Mbps', 'rate 7.556702834472Mbps', 'rate 10.010186476470Mbps', 'rate 10.024873636026Mbps', 'rate 2.037126851851Mbps', 'rate 9.777309213397Mbps', 'rate 2.118682391318Mbps', 'rate 8.405508443070Mbps', 'rate 2.179707922305Mbps', 'rate 9.025860197315Mbps', 'rate 9.967291651117Mbps', 'rate 1.565079026196Mbps', 'rate 2.096041152231Mbps', 'rate 6.299717163743Mbps', 'rate 1.970404015421Mbps', 'rate 7.805785150396Mbps', 'rate 2.434196347026Mbps', 'rate 5.937664099522Mbps', 'rate 9.648548323843Mbps', 'rate 1.539611934988Mbps', 'rate 2.451206945276Mbps', 'rate 1.997100090845Mbps', 'rate 5.290525634575Mbps', 'rate 2.163946417916Mbps', 'rate 8.577954939862Mbps', 'rate 1.839863057405Mbps', 'rate 1.638769705761Mbps', 'rate 8.881191672492Mbps', 'rate 2.293221136042Mbps', 'rate 2.090081685838Mbps', 'rate 2.350372281331Mbps', 'rate 1.955394126662Mbps', 'rate 5.007102117369Mbps', 'rate 8.867457233603Mbps', 'rate 5.317002280795Mbps', 'rate 6.550092574813Mbps', 'rate 9.454339731294Mbps', 'rate 1.917125966251Mbps', 'rate 5.175456324058Mbps', 'rate 2.049570672820Mbps', 'rate 1.652201832558Mbps', 'rate 9.112805251076Mbps', 'rate 2.226173438420Mbps', 'rate 4.590569306196Mbps', 'rate 2.455441121557Mbps', 'rate 6.105905038866Mbps', 'rate 2.397974925468Mbps', 'rate 8.878954248657Mbps', 'rate 10.102798118514Mbps', 'rate 8.627138491099Mbps', 'rate 9.569762470683Mbps', 'rate 8.521226761843Mbps', 'rate 1.799344776381Mbps', 'rate 7.848117282414Mbps', 'rate 1.727159192628Mbps', 'rate 1.650789076820Mbps', 'rate 9.474426231854Mbps', 'rate 1.839719771188Mbps', 'rate 6.002424449819Mbps', 'rate 10.048753596893Mbps', 'rate 5.072777103179Mbps', 'rate 1.717814277372Mbps', 'rate 5.677908126957Mbps', 'rate 1.804807969889Mbps', 'rate 6.080442451505Mbps', 'rate 1.915102025797Mbps', 'rate 9.981521735929Mbps', 'rate 9.918824357530Mbps', 'rate 8.734145410440Mbps', 'rate 1.589201018380Mbps', 'rate 2.065402258752Mbps', 'rate 10.374790831728Mbps', 'rate 7.677213991698Mbps', 'rate 10.285310886621Mbps', 'rate 9.556758955752Mbps', 'rate 8.848864579135Mbps', 'rate 7.851772007950Mbps', 'rate 8.473053654066Mbps', 'rate 2.348025122163Mbps', 'rate 1.825515899824Mbps', 'rate 5.065697226559Mbps', 'rate 4.703900318892Mbps', 'rate 8.992630335431Mbps', 'rate 1.540099591958Mbps', 'rate 6.294822396444Mbps', 'rate 4.956373868121Mbps', 'rate 5.476125991579Mbps', 'rate 5.763792796760Mbps', 'rate 8.655311279445Mbps', 'rate 7.340937188800Mbps', 'rate 1.896617703309Mbps', 'rate 6.799633323381Mbps', 'rate 10.163725753454Mbps', 'rate 9.447527996093Mbps', 'rate 7.397639345722Mbps', 'rate 8.243713794855Mbps', 'rate 1.925530874186Mbps', 'rate 10.243557992420Mbps', 'rate 4.932855052344Mbps', 'rate 1.824731471323Mbps', 'rate 9.090423277497Mbps', 'rate 7.485298162564Mbps', 'rate 6.641532568472Mbps', 'rate 8.366090517493Mbps', 'rate 1.974894080155Mbps', 'rate 2.312064256661Mbps', 'rate 2.020068995770Mbps', 'rate 8.568255668474Mbps', 'rate 9.739150958895Mbps', 'rate 10.083371306598Mbps', 'rate 9.726396225234Mbps', 'rate 2.325324672710Mbps', 'rate 2.422057850570Mbps', 'rate 5.731608739893Mbps', 'rate 8.579561075243Mbps', 'rate 2.140927626200Mbps', 'rate 9.718241624890Mbps', 'rate 6.036321308753Mbps', 'rate 8.901672913901Mbps', 'rate 9.001961774454Mbps', 'rate 7.691992445201Mbps', 'rate 5.547772982982Mbps', 'rate 8.795043547935Mbps', 'rate 9.322764315395Mbps', 'rate 9.111134644413Mbps', 'rate 9.015259116480Mbps', 'rate 10.270214595791Mbps', 'rate 7.459682214548Mbps', 'rate 2.410432995744Mbps', 'rate 2.273498599754Mbps', 'rate 1.585757165581Mbps', 'rate 2.090901533155Mbps', 'rate 7.587005691563Mbps', 'rate 1.527065128062Mbps', 'rate 2.393705156140Mbps', 'rate 7.765937597747Mbps', 'rate 9.515261146246Mbps', 'rate 1.706607228400Mbps', 'rate 9.222281481708Mbps', 'rate 9.164657192281Mbps', 'rate 5.982122211072Mbps', 'rate 6.494304986806Mbps', 'rate 2.260822570899Mbps', 'rate 2.413928624036Mbps', 'rate 7.110758941483Mbps', 'rate 2.473228211569Mbps', 'rate 2.197516202101Mbps', 'rate 1.651797430868Mbps', 'rate 8.888728938346Mbps', 'rate 8.597372851849Mbps', 'rate 8.870767877989Mbps', 'rate 1.858756363312Mbps', 'rate 9.842296400596Mbps', 'rate 4.844916565184Mbps', 'rate 8.948205793793Mbps', 'rate 4.790790095021Mbps', 'rate 9.971024691386Mbps', 'rate 1.500850766278Mbps', 'rate 5.966107031980Mbps', 'rate 1.822624878726Mbps', 'rate 9.892324422885Mbps', 'rate 5.658355873584Mbps', 'rate 9.342593543675Mbps', 'rate 8.618901777449Mbps', 'rate 7.268945187185Mbps', 'rate 9.668419959229Mbps', 'rate 9.506179467075Mbps', 'rate 1.746947616082Mbps', 'rate 1.927814129658Mbps', 'rate 8.930783281161Mbps', 'rate 2.295810890066Mbps', 'rate 5.941104518560Mbps', 'rate 2.231844267089Mbps', 'rate 9.192411616590Mbps', 'rate 4.503028596917Mbps', 'rate 6.782758102486Mbps', 'rate 8.768739844233Mbps', 'rate 4.971787883235Mbps', 'rate 6.545928637416Mbps', 'rate 2.373310149607Mbps', 'rate 5.995372911080Mbps', 'rate 9.766724985156Mbps', 'rate 9.458678601544Mbps', 'rate 6.565694573442Mbps', 'rate 5.240008872724Mbps', 'rate 8.938115708446Mbps', 'rate 9.579095987655Mbps', 'rate 7.197620857815Mbps', 'rate 5.836057575893Mbps', 'rate 10.338944979492Mbps', 'rate 5.355986874243Mbps', 'rate 4.951479362417Mbps', 'rate 4.596212537547Mbps', 'rate 2.079319728599Mbps', 'rate 7.472444995091Mbps', 'rate 8.725048348685Mbps', 'rate 9.742538335406Mbps', 'rate 9.596453039361Mbps', 'rate 2.302982319717Mbps', 'rate 1.506092829404Mbps', 'rate 1.778838452648Mbps', 'rate 4.625653893540Mbps', 'rate 1.998199963003Mbps', 'rate 1.696870666110Mbps', 'rate 1.985982049025Mbps', 'rate 9.677909063722Mbps', 'rate 6.455099886342Mbps', 'rate 10.279135140581Mbps', 'rate 9.933842195680Mbps', 'rate 8.762215958252Mbps', 'rate 6.358587248927Mbps', 'rate 9.904853230745Mbps', 'rate 2.301690585688Mbps', 'rate 7.920240584848Mbps', 'rate 8.283550714606Mbps', 'rate 8.509219812304Mbps', 'rate 10.366335465046Mbps', 'rate 9.437930782251Mbps', 'rate 7.021726232465Mbps', 'rate 9.413587305195Mbps', 'rate 2.277701757267Mbps', 'rate 2.265866288959Mbps', 'rate 1.995554975227Mbps', 'rate 8.825799818666Mbps', 'rate 2.011850422915Mbps', 'rate 9.214937657406Mbps', 'rate 1.949496782060Mbps', 'rate 1.726178362597Mbps', 'rate 2.482677370210Mbps', 'rate 7.809987516368Mbps', 'rate 8.587845995387Mbps', 'rate 1.599555764700Mbps', 'rate 9.442441217576Mbps', 'rate 6.383725085027Mbps', 'rate 9.694855608533Mbps', 'rate 4.542404692938Mbps', 'rate 9.676054136865Mbps', 'rate 9.742678476722Mbps', 'rate 7.808401937253Mbps', 'rate 2.362948594540Mbps', 'rate 10.069957170875Mbps', 'rate 9.928316381598Mbps', 'rate 5.840122129001Mbps', 'rate 1.512278512409Mbps', 'rate 6.425157611840Mbps', 'rate 5.115336620292Mbps', 'rate 2.244956589933Mbps', 'rate 6.107266732004Mbps', 'rate 1.673082648005Mbps', 'rate 9.569956763096Mbps', 'rate 8.651243741086Mbps', 'rate 10.340474146531Mbps', 'rate 5.178762706648Mbps', 'rate 2.059996797111Mbps', 'rate 10.073743030033Mbps', 'rate 9.085629340463Mbps', 'rate 7.303010649833Mbps', 'rate 9.930191382986Mbps', 'rate 6.445615939096Mbps', 'rate 8.732212448923Mbps', 'rate 7.075831781484Mbps', 'rate 2.291153644550Mbps', 'rate 1.506395852123Mbps', 'rate 9.378672491870Mbps', 'rate 9.612768919513Mbps', 'rate 2.037302723271Mbps', 'rate 6.983300790363Mbps', 'rate 2.184539993149Mbps', 'rate 2.044196704429Mbps', 'rate 6.199348672148Mbps', 'rate 4.600305552515Mbps', 'rate 1.612766320674Mbps', 'rate 1.589624371612Mbps', 'rate 5.515080363477Mbps', 'rate 7.739474849017Mbps', 'rate 5.527472477480Mbps', 'rate 8.954217230562Mbps', 'rate 1.672211036240Mbps', 'rate 9.044719809625Mbps', 'rate 6.973360812594Mbps', 'rate 2.376634708541Mbps', 'rate 9.061515041787Mbps', 'rate 5.947962006138Mbps', 'rate 1.536564902322Mbps', 'rate 8.946267007408Mbps', 'rate 10.290787625653Mbps', 'rate 9.272227888737Mbps', 'rate 7.070791881815Mbps', 'rate 6.836013352637Mbps', 'rate 7.157365472717Mbps', 'rate 2.220436523093Mbps', 'rate 1.865882166813Mbps', 'rate 1.898281238681Mbps', 'rate 2.413693203003Mbps', 'rate 4.649052777962Mbps', 'rate 4.518796328272Mbps', 'rate 1.893441067085Mbps', 'rate 10.355676904433Mbps', 'rate 1.858870395468Mbps', 'rate 8.615958667993Mbps', 'rate 9.309266474882Mbps', 'rate 1.548283427176Mbps', 'rate 9.766037235701Mbps', 'rate 6.156017899844Mbps', 'rate 1.975680252551Mbps', 'rate 8.161556749362Mbps', 'rate 2.276033876788Mbps', 'rate 2.169021013650Mbps', 'rate 9.850965022893Mbps', 'rate 8.542056658942Mbps', 'rate 8.334942117135Mbps', 'rate 1.649268402355Mbps', 'rate 7.518729052705Mbps', 'rate 2.033923027009Mbps', 'rate 7.063318384068Mbps', 'rate 1.650195331651Mbps', 'rate 6.141122388760Mbps', 'rate 10.279191116216Mbps', 'rate 5.142995292249Mbps', 'rate 2.411181062025Mbps', 'rate 1.998645677569Mbps', 'rate 1.583523476842Mbps', 'rate 8.685894071834Mbps', 'rate 2.243674799571Mbps', 'rate 5.754324540745Mbps', 'rate 8.606344191517Mbps', 'rate 1.804422406782Mbps', 'rate 5.713766378050Mbps', 'rate 9.059677874335Mbps', 'rate 8.979252823723Mbps', 'rate 6.957251916894Mbps', 'rate 2.090282184778Mbps', 'rate 9.290762895816Mbps', 'rate 2.215556365340Mbps', 'rate 4.868792689628Mbps', 'rate 10.002604522915Mbps', 'rate 9.420423832302Mbps', 'rate 2.301706661629Mbps', 'rate 9.700837762079Mbps', 'rate 6.127987007544Mbps', 'rate 5.996213024101Mbps', 'rate 1.983763124761Mbps', 'rate 7.104060108265Mbps', 'rate 2.283274174865Mbps', 'rate 5.906359916327Mbps', 'rate 5.448240913714Mbps', 'rate 1.592183863977Mbps', 'rate 10.271878028188Mbps', 'rate 7.978024829700Mbps', 'rate 2.183510248285Mbps', 'rate 10.291942463414Mbps', 'rate 9.080879976307Mbps', 'rate 9.207234396793Mbps', 'rate 6.996002339270Mbps', 'rate 10.270346841060Mbps', 'rate 10.073947570241Mbps', 'rate 4.915053595343Mbps', 'rate 6.822971762834Mbps', 'rate 9.661616869293Mbps', 'rate 9.342123284766Mbps', 'rate 9.834320215352Mbps', 'rate 1.658367293582Mbps', 'rate 7.064152196388Mbps', 'rate 6.956341082986Mbps', 'rate 2.091586799854Mbps', 'rate 9.997691357122Mbps', 'rate 2.305684759677Mbps', 'rate 4.611709493097Mbps', 'rate 10.174899394575Mbps', 'rate 9.861497653577Mbps', 'rate 4.979491102910Mbps', 'rate 9.887587015382Mbps', 'rate 9.902670857796Mbps', 'rate 7.101510459242Mbps', 'rate 8.022247833691Mbps', 'rate 7.944820402109Mbps', 'rate 9.758670906950Mbps', 'rate 10.348386396846Mbps', 'rate 9.996675434493Mbps', 'rate 6.200299676791Mbps', 'rate 5.398653143125Mbps', 'rate 7.735825671644Mbps', 'rate 4.981564436466Mbps', 'rate 8.831971434176Mbps', 'rate 2.486662677187Mbps', 'rate 10.166350355680Mbps', 'rate 2.443880030745Mbps', 'rate 2.030362219523Mbps', 'rate 2.225494408063Mbps', 'rate 2.029531475697Mbps', 'rate 10.335854231761Mbps', 'rate 2.425770053720Mbps', 'rate 9.082522726079Mbps', 'rate 8.817339191642Mbps', 'rate 1.687601855313Mbps', 'rate 2.078444881221Mbps', 'rate 1.804504013248Mbps', 'rate 9.873305632584Mbps', 'rate 9.001466174271Mbps', 'rate 1.837131478240Mbps', 'rate 2.113471250311Mbps', 'rate 5.196349456359Mbps', 'rate 6.826888745334Mbps', 'rate 1.758079834409Mbps', 'rate 10.200179583009Mbps', 'rate 1.615438227674Mbps', 'rate 10.235623807014Mbps', 'rate 4.873330240091Mbps', 'rate 8.924969882028Mbps', 'rate 7.684504474646Mbps', 'rate 5.177308636281Mbps', 'rate 9.349045595806Mbps', 'rate 5.487864875093Mbps', 'rate 5.452720890959Mbps', 'rate 9.184455507903Mbps', 'rate 2.136828197398Mbps', 'rate 4.553764184093Mbps', 'rate 9.552308047815Mbps', 'rate 7.308512742884Mbps', 'rate 2.399629780261Mbps', 'rate 9.202493534803Mbps', 'rate 4.924235641640Mbps', 'rate 9.991744774936Mbps', 'rate 7.677002957332Mbps', 'rate 5.182933848514Mbps', 'rate 9.448404757952Mbps', 'rate 7.478279291260Mbps', 'rate 1.620141205751Mbps', 'rate 9.905446625925Mbps', 'rate 1.914112724338Mbps', 'rate 2.339233008398Mbps', 'rate 7.509903978128Mbps', 'rate 2.321211547413Mbps', 'rate 1.816032295405Mbps', 'rate 1.953309689083Mbps', 'rate 1.908526188193Mbps', 'rate 5.074527848507Mbps', 'rate 10.156533304824Mbps', 'rate 6.727460471071Mbps', 'rate 2.427046000453Mbps', 'rate 1.963023580485Mbps', 'rate 8.950525497134Mbps', 'rate 9.018960789443Mbps', 'rate 9.531558876581Mbps', 'rate 9.375128444427Mbps', 'rate 8.660323539610Mbps', 'rate 9.523724784980Mbps', 'rate 2.280859515613Mbps', 'rate 7.309589968598Mbps', 'rate 9.285875735077Mbps', 'rate 2.036317253939Mbps', 'rate 9.154663495637Mbps', 'rate 1.568760113454Mbps', 'rate 9.840372483253Mbps', 'rate 6.678432490075Mbps', 'rate 9.774990585314Mbps', 'rate 1.717410177488Mbps', 'rate 6.641994262826Mbps', 'rate 8.717570026129Mbps', 'rate 2.127218070526Mbps', 'rate 10.102468358051Mbps', 'rate 9.689693904020Mbps', 'rate 2.365255553691Mbps', 'rate 7.046992813986Mbps', 'rate 8.527344315568Mbps', 'rate 9.191947451322Mbps', 'rate 2.032648657678Mbps', 'rate 10.066701161740Mbps', 'rate 9.175351608141Mbps', 'rate 5.818403530380Mbps', 'rate 10.261531237907Mbps', 'rate 9.568409571240Mbps', 'rate 6.329661613442Mbps', 'rate 9.456873720784Mbps', 'rate 9.516514114657Mbps', 'rate 8.542823948617Mbps', 'rate 8.860275175643Mbps', 'rate 9.017874993950Mbps', 'rate 5.291433893411Mbps', 'rate 2.149279023073Mbps', 'rate 10.049294595172Mbps', 'rate 5.378148657681Mbps', 'rate 6.267698935742Mbps', 'rate 2.476431724794Mbps', 'rate 9.819219764476Mbps', 'rate 1.500498750536Mbps', 'rate 2.496646977448Mbps', 'rate 1.641895769796Mbps', 'rate 8.344021279834Mbps', 'rate 1.658796246516Mbps', 'rate 8.943015491238Mbps', 'rate 2.400767336874Mbps', 'rate 8.981114192937Mbps', 'rate 8.874563210161Mbps', 'rate 1.749220156698Mbps', 'rate 9.830682329098Mbps']
        
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
