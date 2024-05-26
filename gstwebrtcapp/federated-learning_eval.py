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

MODEL_FILE = f"fedModelsReward/fixed_rRate = 0.25 1 Agent (crash 220)/drl_model_11000_steps.zip"


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
        network_controller.rules = ['rate 8.795984749726982Mbps', 'rate 9.39350982946294Mbps', 'rate 8.749737799604514Mbps', 'rate 8.366684073629493Mbps', 'rate 6.8300549876256Mbps', 'rate 1.9549092836725344Mbps', 'rate 8.667162173636628Mbps', 'rate 8.83917409516552Mbps', 'rate 8.313910955652545Mbps', 'rate 8.902768242888307Mbps', 'rate 9.858967476581444Mbps', 'rate 1.6521514512358824Mbps', 'rate 1.3975604828956945Mbps', 'rate 8.867875691681233Mbps', 'rate 1.462503489972856Mbps', 'rate 4.825626994975102Mbps', 'rate 9.530672988981403Mbps', 'rate 9.457762597168387Mbps', 'rate 1.0490785217877994Mbps', 'rate 8.739065774439073Mbps', 'rate 1.8899036669074145Mbps', 'rate 7.367391080155286Mbps', 'rate 1.824306080929953Mbps', 'rate 1.192685797840414Mbps', 'rate 1.1860417160049361Mbps', 'rate 8.40319626272118Mbps', 'rate 1.703561894297796Mbps', 'rate 1.302443442074931Mbps', 'rate 8.201513707427742Mbps', 'rate 1.410814029226064Mbps', 'rate 8.558720176662174Mbps', 'rate 1.0217908317550615Mbps', 'rate 1.0774853658211603Mbps', 'rate 1.9712773739482647Mbps', 'rate 9.235678956228991Mbps', 'rate 8.982674951671175Mbps', 'rate 9.787526849219022Mbps', 'rate 7.507232370023594Mbps', 'rate 1.4015232565951008Mbps', 'rate 8.654496096526175Mbps', 'rate 1.3407676986883934Mbps', 'rate 1.628600390916138Mbps', 'rate 9.684440600924678Mbps', 'rate 9.557353285957722Mbps', 'rate 8.254508721842884Mbps', 'rate 8.054247242835745Mbps', 'rate 4.832785209819818Mbps', 'rate 1.572438930118063Mbps', 'rate 1.8265509511484015Mbps', 'rate 1.4399762908230747Mbps', 'rate 1.9162181271799565Mbps', 'rate 6.39175316831413Mbps', 'rate 8.88319404660438Mbps', 'rate 1.2026754753216515Mbps', 'rate 8.293378041323486Mbps', 'rate 9.38196754046996Mbps', 'rate 9.767159116774472Mbps', 'rate 8.098097858454036Mbps', 'rate 6.659052258403973Mbps', 'rate 4.725635145466164Mbps', 'rate 5.254192568661157Mbps', 'rate 8.065587353074461Mbps', 'rate 9.703997866125093Mbps', 'rate 7.125193226738822Mbps', 'rate 4.392147723329721Mbps', 'rate 1.9245925230932932Mbps', 'rate 8.501935980495192Mbps', 'rate 1.749493367219767Mbps', 'rate 1.6880842127763493Mbps', 'rate 6.648800539510772Mbps', 'rate 4.119569963432969Mbps', 'rate 8.455362317099857Mbps', 'rate 1.43009855252384Mbps', 'rate 1.0785017109988266Mbps', 'rate 6.708403239637492Mbps', 'rate 1.1888098708441175Mbps', 'rate 4.943017781429686Mbps', 'rate 9.82781034771452Mbps', 'rate 7.788337357612923Mbps', 'rate 8.62401409968233Mbps', 'rate 8.463318483248504Mbps', 'rate 5.518487648252354Mbps', 'rate 4.167734732736747Mbps', 'rate 1.3288908286010548Mbps', 'rate 5.562283742981807Mbps', 'rate 1.6027635974421117Mbps', 'rate 1.5230956381742162Mbps', 'rate 1.0314986672197193Mbps', 'rate 5.708643676837342Mbps', 'rate 5.99302001055621Mbps', 'rate 9.889148702789642Mbps', 'rate 8.458698493693902Mbps', 'rate 4.650944541241525Mbps', 'rate 9.236465503076648Mbps', 'rate 6.875177619952604Mbps', 'rate 7.879370638495789Mbps', 'rate 9.222464168270369Mbps', 'rate 1.7447414210642225Mbps', 'rate 6.090697274489232Mbps', 'rate 8.505221764331969Mbps', 'rate 8.371647338876667Mbps', 'rate 1.4844151504749288Mbps', 'rate 5.964369635364175Mbps', 'rate 4.800441367605059Mbps', 'rate 1.6709546663442372Mbps', 'rate 9.64703438281577Mbps', 'rate 1.806283213746923Mbps', 'rate 7.930038022272594Mbps', 'rate 9.133182916147371Mbps', 'rate 4.302926085976425Mbps', 'rate 1.8253434836256814Mbps', 'rate 7.996045036764453Mbps', 'rate 5.1600981403385795Mbps', 'rate 9.867979104536122Mbps', 'rate 9.143047577338207Mbps', 'rate 9.384863293038926Mbps', 'rate 6.16662362594799Mbps', 'rate 5.238636737571207Mbps', 'rate 9.086712568373105Mbps', 'rate 1.1727753854106613Mbps', 'rate 1.947050045980822Mbps', 'rate 8.38960539981329Mbps', 'rate 9.815968301091514Mbps', 'rate 9.78955527154259Mbps', 'rate 1.4496548338474744Mbps', 'rate 8.057470201150648Mbps', 'rate 9.347087397105128Mbps', 'rate 1.5866609905280689Mbps', 'rate 8.495833268243016Mbps', 'rate 1.8927823844792866Mbps', 'rate 4.73978656309721Mbps', 'rate 5.279636438893361Mbps', 'rate 8.666043738254432Mbps', 'rate 1.1538662960491681Mbps', 'rate 7.792008346837992Mbps', 'rate 8.769275696433963Mbps', 'rate 8.238424522599482Mbps', 'rate 1.9885717240096712Mbps', 'rate 8.054914042943656Mbps', 'rate 6.707096254643247Mbps', 'rate 5.290535340543137Mbps', 'rate 8.25386710238094Mbps', 'rate 8.859926692250967Mbps', 'rate 8.29563250080286Mbps', 'rate 9.347709535635216Mbps', 'rate 9.301383995484873Mbps', 'rate 7.546866765474936Mbps', 'rate 9.592512442633407Mbps', 'rate 8.122890291869567Mbps', 'rate 9.60932044870484Mbps', 'rate 1.38300209697235Mbps', 'rate 1.1301746731841666Mbps', 'rate 1.9264549962328261Mbps', 'rate 9.228980752668082Mbps', 'rate 1.123905013229651Mbps', 'rate 8.646581391424208Mbps', 'rate 5.845064542950112Mbps', 'rate 5.145785581381943Mbps', 'rate 4.045739947892274Mbps', 'rate 1.7241688616307136Mbps', 'rate 8.029260496138301Mbps', 'rate 1.549520905372967Mbps', 'rate 8.796563010572864Mbps', 'rate 6.510484506158132Mbps', 'rate 9.0618627923725Mbps', 'rate 9.8285725307746Mbps', 'rate 8.33506734679754Mbps', 'rate 1.5316961631980472Mbps', 'rate 9.224213403357938Mbps', 'rate 9.164558212015574Mbps', 'rate 1.1258780298341087Mbps', 'rate 5.790292307980122Mbps', 'rate 6.858789329796425Mbps', 'rate 9.490272633598211Mbps', 'rate 6.433141706962932Mbps', 'rate 8.269337685915673Mbps', 'rate 9.624549045356169Mbps', 'rate 9.061375196983112Mbps', 'rate 5.159444897420536Mbps', 'rate 5.367620270678064Mbps', 'rate 1.352656134328365Mbps', 'rate 7.360452723024417Mbps', 'rate 8.429823491150767Mbps', 'rate 1.7972893098753362Mbps', 'rate 1.7355845242737182Mbps', 'rate 4.642539033097224Mbps', 'rate 1.012201500514493Mbps', 'rate 8.923998277911494Mbps', 'rate 6.649523941422087Mbps', 'rate 7.445591782132105Mbps', 'rate 1.0632958185488774Mbps', 'rate 9.854722229801883Mbps', 'rate 1.6683843271484942Mbps', 'rate 7.840529825056885Mbps', 'rate 8.439003849721908Mbps', 'rate 8.896543290575496Mbps', 'rate 1.010154294646291Mbps', 'rate 1.0726631056165017Mbps', 'rate 8.639500167379252Mbps', 'rate 8.285059455539528Mbps', 'rate 5.720640298315443Mbps', 'rate 9.301686374396906Mbps', 'rate 5.2980489407372815Mbps', 'rate 6.617078604746553Mbps', 'rate 9.120908695312934Mbps', 'rate 8.257359908920293Mbps', 'rate 5.374320149638702Mbps', 'rate 4.53790778694923Mbps', 'rate 4.015217663014402Mbps', 'rate 8.02875872417893Mbps', 'rate 4.7439052644662025Mbps', 'rate 4.302073439723932Mbps', 'rate 6.580244534077069Mbps', 'rate 1.3948082233645356Mbps', 'rate 9.731559190047713Mbps', 'rate 1.3334109854280132Mbps', 'rate 8.636944547324704Mbps', 'rate 8.865548069706207Mbps', 'rate 5.8153396220779685Mbps', 'rate 5.1822714198496325Mbps', 'rate 7.846485947747175Mbps', 'rate 8.940959036678171Mbps', 'rate 4.385145888144232Mbps', 'rate 6.417646694945926Mbps', 'rate 8.69710220039406Mbps', 'rate 1.663110050233524Mbps', 'rate 1.0502418965768414Mbps', 'rate 8.398309160799691Mbps', 'rate 8.889061392625177Mbps', 'rate 9.742348586311605Mbps', 'rate 8.774234838507915Mbps', 'rate 1.4325452899591624Mbps', 'rate 8.084603011157737Mbps', 'rate 8.56315948758475Mbps', 'rate 8.480572713751098Mbps', 'rate 1.6108335791302453Mbps', 'rate 8.833647512440331Mbps', 'rate 7.297490368246477Mbps', 'rate 1.685971724408048Mbps', 'rate 8.531510423926202Mbps', 'rate 6.551742069702518Mbps', 'rate 4.39781277057281Mbps', 'rate 9.262826314298394Mbps', 'rate 4.087401978993746Mbps', 'rate 1.6143832403503704Mbps', 'rate 1.141850517559631Mbps', 'rate 8.163774945610204Mbps', 'rate 8.452219517576085Mbps', 'rate 8.173180948438516Mbps', 'rate 1.3343228957652515Mbps', 'rate 5.788861733773558Mbps', 'rate 8.013365036754521Mbps', 'rate 9.716199944103106Mbps', 'rate 1.0316817122451087Mbps', 'rate 9.722711808214996Mbps', 'rate 8.906437546268608Mbps', 'rate 1.1774408722057916Mbps', 'rate 4.322267660306488Mbps', 'rate 9.875273567703314Mbps', 'rate 1.1917430390111432Mbps', 'rate 8.518116282915653Mbps', 'rate 9.453621536460433Mbps', 'rate 6.372240024791164Mbps', 'rate 4.825464990244431Mbps', 'rate 1.622729977311233Mbps', 'rate 8.527687091323001Mbps', 'rate 1.4168192126029509Mbps', 'rate 1.3639734488935789Mbps', 'rate 9.89475632633514Mbps', 'rate 7.021674387124186Mbps', 'rate 1.0380726529367719Mbps', 'rate 1.6549807983821716Mbps', 'rate 8.350827091684604Mbps', 'rate 8.91315753784602Mbps', 'rate 1.0519234849282495Mbps', 'rate 1.9743547600748204Mbps', 'rate 1.0449221658274532Mbps', 'rate 8.476374580144471Mbps', 'rate 7.536317248862408Mbps', 'rate 4.7210830313618075Mbps', 'rate 6.933127900990406Mbps', 'rate 1.4636168961371006Mbps', 'rate 9.85008015083538Mbps', 'rate 1.8986236469984967Mbps', 'rate 1.7146083532752476Mbps', 'rate 8.286261572930165Mbps', 'rate 1.7637610880864405Mbps', 'rate 6.457813772735282Mbps', 'rate 9.258168177564635Mbps', 'rate 1.7073336855998797Mbps', 'rate 1.7359203263519738Mbps', 'rate 5.086811044128554Mbps', 'rate 1.3833579354073828Mbps', 'rate 5.467640114074729Mbps', 'rate 9.1305165568668Mbps', 'rate 1.7532604043152622Mbps', 'rate 9.760005937585262Mbps', 'rate 1.3092402199215845Mbps', 'rate 8.52322652816433Mbps', 'rate 5.284058993979963Mbps', 'rate 9.526980262540862Mbps', 'rate 4.353014523818924Mbps', 'rate 9.127729304638056Mbps', 'rate 6.650046494479822Mbps', 'rate 5.6545549514356495Mbps', 'rate 6.099660427572193Mbps', 'rate 7.51553370421434Mbps', 'rate 8.605074056625583Mbps', 'rate 9.126228508509659Mbps', 'rate 1.8918208882449794Mbps', 'rate 6.449557675321103Mbps', 'rate 1.8987355771137016Mbps', 'rate 9.380801538842574Mbps', 'rate 8.524397403435767Mbps', 'rate 1.704112399657768Mbps', 'rate 8.354982951124743Mbps', 'rate 9.123502228066345Mbps', 'rate 9.76411922487181Mbps', 'rate 8.810431451061477Mbps', 'rate 4.093196876056243Mbps', 'rate 7.9115040082568555Mbps', 'rate 1.6433782537734962Mbps', 'rate 7.26992152166758Mbps', 'rate 5.856001442783851Mbps', 'rate 5.710440933074478Mbps', 'rate 9.268907304608106Mbps', 'rate 1.1064458984475138Mbps', 'rate 8.510295529378261Mbps', 'rate 8.562360948002233Mbps', 'rate 1.5443782140977467Mbps', 'rate 1.1789147940882752Mbps', 'rate 8.322366164964897Mbps', 'rate 1.9501826398668158Mbps', 'rate 4.4998495617985155Mbps', 'rate 4.1275633052677865Mbps', 'rate 1.5478539660196775Mbps', 'rate 8.027873119655933Mbps', 'rate 7.381513392649155Mbps', 'rate 8.707865733728813Mbps', 'rate 1.7360990299611343Mbps', 'rate 1.0784660658610181Mbps', 'rate 5.149772382553454Mbps', 'rate 8.0352552003358Mbps', 'rate 8.159206594001823Mbps', 'rate 7.551674324759306Mbps', 'rate 1.6870443100563237Mbps', 'rate 9.239663435714625Mbps', 'rate 5.331600613219071Mbps', 'rate 4.435426533312091Mbps', 'rate 1.1293845183653102Mbps', 'rate 6.950710506274344Mbps', 'rate 1.0586341515478743Mbps', 'rate 5.872463242322841Mbps', 'rate 1.167103381526072Mbps', 'rate 1.6101698241336946Mbps', 'rate 8.527750929372528Mbps', 'rate 1.0823423619446118Mbps', 'rate 9.875737195146685Mbps', 'rate 8.632027704965498Mbps', 'rate 8.68457786742246Mbps', 'rate 8.125367339146505Mbps', 'rate 6.419088497572387Mbps', 'rate 8.660147349428922Mbps', 'rate 8.177304099659237Mbps', 'rate 8.481911698074038Mbps', 'rate 9.170139927154745Mbps', 'rate 5.644611190329845Mbps', 'rate 8.87729867318827Mbps', 'rate 7.135243652695215Mbps', 'rate 7.996424391586736Mbps', 'rate 9.422668507478633Mbps', 'rate 8.521873360257645Mbps', 'rate 1.7920363238005341Mbps', 'rate 8.007072708908401Mbps', 'rate 4.694526077474487Mbps', 'rate 8.154063175294825Mbps', 'rate 7.940148884511619Mbps', 'rate 1.0775193754862789Mbps', 'rate 9.179169241251746Mbps', 'rate 9.871771446066072Mbps', 'rate 1.6497332293565896Mbps', 'rate 9.604771798791436Mbps', 'rate 7.873985367994402Mbps', 'rate 9.15774603634768Mbps', 'rate 7.884360120858009Mbps', 'rate 1.7055776978032298Mbps', 'rate 1.3197085194426095Mbps', 'rate 5.31675349585992Mbps', 'rate 9.642622676891799Mbps', 'rate 8.955522851720513Mbps', 'rate 8.707983502066673Mbps', 'rate 1.754626767335989Mbps', 'rate 1.178635920104726Mbps', 'rate 9.020759351142262Mbps', 'rate 6.348418299616563Mbps', 'rate 7.545498614925085Mbps', 'rate 6.7499584687246275Mbps', 'rate 8.460990636290013Mbps', 'rate 1.329374484123798Mbps', 'rate 8.457811132792507Mbps', 'rate 5.467192175334692Mbps', 'rate 9.326667004251263Mbps', 'rate 4.406409850653084Mbps', 'rate 9.843893875858942Mbps', 'rate 8.941531932679817Mbps', 'rate 1.690600552145531Mbps', 'rate 7.577965512902786Mbps', 'rate 4.939266839504981Mbps', 'rate 1.5209868336133163Mbps', 'rate 5.867148810664755Mbps', 'rate 8.472149337529066Mbps', 'rate 8.230820652924441Mbps', 'rate 8.532175899294629Mbps', 'rate 7.919914047453766Mbps', 'rate 1.3812899585689897Mbps', 'rate 9.201949182299343Mbps', 'rate 1.2019419408874203Mbps', 'rate 1.684267398085661Mbps', 'rate 9.221920060758185Mbps', 'rate 9.896174433092128Mbps', 'rate 1.0598847836639957Mbps', 'rate 8.110532615474474Mbps', 'rate 1.1048037965048314Mbps', 'rate 5.347397501454173Mbps', 'rate 9.69178994596917Mbps', 'rate 8.514071332266042Mbps', 'rate 1.1335226575085413Mbps', 'rate 5.066663927204523Mbps', 'rate 6.970500614754496Mbps', 'rate 9.134236777104029Mbps', 'rate 8.3490403021626Mbps', 'rate 1.7920762364906329Mbps', 'rate 8.395079609119776Mbps', 'rate 1.2530157798381152Mbps', 'rate 8.234388433865668Mbps', 'rate 8.263425214363814Mbps', 'rate 8.482558872700992Mbps', 'rate 8.738452466677305Mbps', 'rate 1.158527042426885Mbps', 'rate 1.744826185874592Mbps', 'rate 1.6916184720381482Mbps', 'rate 8.681320821426553Mbps', 'rate 5.518627328922154Mbps', 'rate 6.733307297992223Mbps', 'rate 1.9514590421075844Mbps', 'rate 7.899484839727148Mbps', 'rate 6.726264017591353Mbps', 'rate 4.117693430738045Mbps', 'rate 7.874578004111804Mbps', 'rate 9.101603318703035Mbps', 'rate 1.0138837134763534Mbps', 'rate 9.719890390973614Mbps', 'rate 1.4952303241138982Mbps', 'rate 4.050230191227989Mbps', 'rate 5.426016732539946Mbps', 'rate 8.86415517009762Mbps', 'rate 1.795008974785178Mbps', 'rate 4.849173531600666Mbps', 'rate 9.802224696759046Mbps', 'rate 9.606445649157601Mbps', 'rate 8.864933873481974Mbps', 'rate 1.5490778853733866Mbps', 'rate 9.788368769383025Mbps', 'rate 8.078395951019044Mbps', 'rate 7.929501496813658Mbps', 'rate 1.482530338013738Mbps', 'rate 4.920509680696897Mbps', 'rate 9.119101851679046Mbps', 'rate 1.4762508287778418Mbps', 'rate 8.857048248677739Mbps', 'rate 9.50265569052233Mbps', 'rate 1.863740644092616Mbps', 'rate 1.8525008445783782Mbps', 'rate 7.428424274015734Mbps', 'rate 1.358787136213806Mbps', 'rate 1.958741494003161Mbps', 'rate 1.0732922988717304Mbps', 'rate 9.78228896086275Mbps', 'rate 8.375057804901791Mbps', 'rate 5.0407840497638805Mbps', 'rate 8.435475262919802Mbps', 'rate 1.0150351640773287Mbps', 'rate 8.937874102684512Mbps', 'rate 1.222109739770712Mbps', 'rate 9.779444044118572Mbps', 'rate 1.570265217716308Mbps', 'rate 6.712480236944097Mbps', 'rate 9.54506463130903Mbps', 'rate 8.824715062332242Mbps', 'rate 4.323268102105072Mbps', 'rate 4.012005256039066Mbps', 'rate 8.018672251462105Mbps', 'rate 9.545065476525235Mbps', 'rate 1.7568223054735346Mbps', 'rate 8.950907200752198Mbps', 'rate 1.5375944288664556Mbps', 'rate 7.226484417365871Mbps', 'rate 1.5795748148408881Mbps', 'rate 4.234665477197572Mbps', 'rate 1.4209180544028084Mbps', 'rate 5.763209837594271Mbps', 'rate 1.309557478516668Mbps', 'rate 9.121444173047207Mbps', 'rate 4.772817817042199Mbps', 'rate 8.931264716565273Mbps', 'rate 7.05670283447248Mbps', 'rate 9.510186476470123Mbps', 'rate 9.524873636026427Mbps', 'rate 1.5371268518514307Mbps', 'rate 9.277309213396716Mbps', 'rate 1.6186823913179562Mbps', 'rate 7.905508443069779Mbps', 'rate 1.6797079223050295Mbps', 'rate 8.525860197315177Mbps', 'rate 9.467291651116806Mbps', 'rate 1.065079026195729Mbps', 'rate 1.5960411522311588Mbps', 'rate 5.799717163742724Mbps', 'rate 1.4704040154208993Mbps', 'rate 7.305785150396041Mbps', 'rate 1.9341963470263435Mbps', 'rate 5.437664099521552Mbps', 'rate 9.14854832384281Mbps', 'rate 1.039611934988499Mbps', 'rate 1.9512069452763399Mbps', 'rate 1.4971000908446936Mbps', 'rate 4.790525634574712Mbps', 'rate 1.6639464179161043Mbps', 'rate 8.077954939862419Mbps', 'rate 1.339863057405108Mbps', 'rate 1.1387697057609836Mbps', 'rate 8.381191672491665Mbps', 'rate 1.7932211360420653Mbps', 'rate 1.5900816858378315Mbps', 'rate 1.850372281330526Mbps', 'rate 1.4553941266619264Mbps', 'rate 4.507102117369174Mbps', 'rate 8.367457233602961Mbps', 'rate 4.817002280794981Mbps', 'rate 6.050092574812796Mbps', 'rate 8.95433973129371Mbps', 'rate 1.4171259662508513Mbps', 'rate 4.675456324057876Mbps', 'rate 1.5495706728200607Mbps', 'rate 1.1522018325583852Mbps', 'rate 8.612805251075963Mbps', 'rate 1.7261734384200351Mbps', 'rate 4.090569306196196Mbps', 'rate 1.9554411215571705Mbps', 'rate 5.605905038866283Mbps', 'rate 1.8979749254680311Mbps', 'rate 8.378954248657045Mbps', 'rate 9.60279811851442Mbps', 'rate 8.127138491098833Mbps', 'rate 9.06976247068307Mbps', 'rate 8.021226761842907Mbps', 'rate 1.2993447763810204Mbps', 'rate 7.348117282414407Mbps', 'rate 1.227159192628244Mbps', 'rate 1.1507890768200104Mbps', 'rate 8.974426231854228Mbps', 'rate 1.3397197711881201Mbps', 'rate 5.50242444981873Mbps', 'rate 9.54875359689279Mbps', 'rate 4.572777103179254Mbps', 'rate 1.217814277372009Mbps', 'rate 5.177908126956641Mbps', 'rate 1.3048079698891886Mbps', 'rate 5.580442451504522Mbps', 'rate 1.4151020257971014Mbps', 'rate 9.4815217359292Mbps', 'rate 9.418824357530275Mbps', 'rate 8.23414541044042Mbps', 'rate 1.0892010183803609Mbps', 'rate 1.565402258752406Mbps', 'rate 9.874790831727791Mbps', 'rate 7.177213991698304Mbps', 'rate 9.785310886620595Mbps', 'rate 9.05675895575225Mbps', 'rate 8.348864579134654Mbps', 'rate 7.351772007950284Mbps', 'rate 7.973053654066127Mbps', 'rate 1.848025122163003Mbps', 'rate 1.325515899824223Mbps', 'rate 4.565697226559464Mbps', 'rate 4.2039003188923765Mbps', 'rate 8.492630335431286Mbps', 'rate 1.0400995919576128Mbps', 'rate 5.79482239644398Mbps', 'rate 4.456373868120804Mbps', 'rate 4.976125991579476Mbps', 'rate 5.263792796759985Mbps', 'rate 8.155311279444863Mbps', 'rate 6.840937188799774Mbps', 'rate 1.3966177033094032Mbps', 'rate 6.299633323381357Mbps', 'rate 9.663725753453889Mbps', 'rate 8.947527996092578Mbps', 'rate 6.897639345722443Mbps', 'rate 7.743713794855044Mbps', 'rate 1.4255308741863078Mbps', 'rate 9.743557992419735Mbps', 'rate 4.432855052343783Mbps', 'rate 1.3247314713229708Mbps', 'rate 8.59042327749749Mbps', 'rate 6.985298162563916Mbps', 'rate 6.141532568472459Mbps', 'rate 7.866090517493237Mbps', 'rate 1.4748940801547747Mbps', 'rate 1.8120642566612526Mbps', 'rate 1.5200689957702376Mbps', 'rate 8.068255668474093Mbps', 'rate 9.239150958895003Mbps', 'rate 9.583371306597769Mbps', 'rate 9.226396225234113Mbps', 'rate 1.8253246727100556Mbps', 'rate 1.922057850569641Mbps', 'rate 5.2316087398931215Mbps', 'rate 8.079561075242804Mbps', 'rate 1.6409276261998116Mbps', 'rate 9.218241624890286Mbps', 'rate 5.536321308752992Mbps', 'rate 8.40167291390103Mbps', 'rate 8.501961774454406Mbps', 'rate 7.19199244520088Mbps', 'rate 5.047772982982048Mbps', 'rate 8.295043547934995Mbps', 'rate 8.822764315395023Mbps', 'rate 8.611134644413426Mbps', 'rate 8.51525911647983Mbps', 'rate 9.770214595790733Mbps', 'rate 6.959682214547824Mbps', 'rate 1.9104329957440989Mbps', 'rate 1.7734985997540698Mbps', 'rate 1.0857571655811182Mbps', 'rate 1.590901533155411Mbps', 'rate 7.08700569156322Mbps', 'rate 1.0270651280620773Mbps', 'rate 1.8937051561395097Mbps', 'rate 7.265937597746754Mbps', 'rate 9.01526114624616Mbps', 'rate 1.20660722839987Mbps', 'rate 8.72228148170778Mbps', 'rate 8.664657192280966Mbps', 'rate 5.4821222110720464Mbps', 'rate 5.994304986805774Mbps', 'rate 1.7608225708990455Mbps', 'rate 1.9139286240359037Mbps', 'rate 6.610758941483375Mbps', 'rate 1.9732282115685815Mbps', 'rate 1.6975162021011472Mbps', 'rate 1.1517974308682257Mbps', 'rate 8.388728938345768Mbps', 'rate 8.097372851848595Mbps', 'rate 8.370767877988612Mbps', 'rate 1.3587563633124837Mbps', 'rate 9.342296400596254Mbps', 'rate 4.344916565184274Mbps', 'rate 8.448205793792972Mbps', 'rate 4.290790095020622Mbps', 'rate 9.47102469138618Mbps', 'rate 1.0008507662778396Mbps', 'rate 5.46610703197962Mbps', 'rate 1.3226248787264532Mbps', 'rate 9.392324422884965Mbps', 'rate 5.158355873584211Mbps', 'rate 8.842593543675427Mbps', 'rate 8.118901777448901Mbps', 'rate 6.7689451871851425Mbps', 'rate 9.168419959228643Mbps', 'rate 9.00617946707482Mbps', 'rate 1.2469476160817696Mbps', 'rate 1.4278141296582612Mbps', 'rate 8.430783281160743Mbps', 'rate 1.7958108900663803Mbps', 'rate 5.441104518560126Mbps', 'rate 1.7318442670892875Mbps', 'rate 8.692411616589697Mbps', 'rate 4.003028596916986Mbps', 'rate 6.282758102485864Mbps', 'rate 8.268739844233492Mbps', 'rate 4.4717878832346Mbps', 'rate 6.045928637415533Mbps', 'rate 1.8733101496067976Mbps', 'rate 5.49537291108002Mbps', 'rate 9.266724985156365Mbps', 'rate 8.958678601543804Mbps', 'rate 6.065694573442036Mbps', 'rate 4.740008872723617Mbps', 'rate 8.438115708446178Mbps', 'rate 9.079095987654632Mbps', 'rate 6.697620857815392Mbps', 'rate 5.336057575893367Mbps', 'rate 9.838944979492142Mbps', 'rate 4.85598687424343Mbps', 'rate 4.4514793624172295Mbps', 'rate 4.096212537546952Mbps', 'rate 1.5793197285986564Mbps', 'rate 6.972444995091162Mbps', 'rate 8.225048348684783Mbps', 'rate 9.242538335405987Mbps', 'rate 9.096453039360586Mbps', 'rate 1.802982319717275Mbps', 'rate 1.006092829403722Mbps', 'rate 1.2788384526478311Mbps', 'rate 4.12565389354002Mbps', 'rate 1.4981999630030154Mbps', 'rate 1.1968706661095738Mbps', 'rate 1.4859820490246447Mbps', 'rate 9.177909063721835Mbps', 'rate 5.955099886342337Mbps', 'rate 9.779135140580566Mbps', 'rate 9.433842195679812Mbps', 'rate 8.26221595825165Mbps', 'rate 5.858587248927179Mbps', 'rate 9.40485323074477Mbps', 'rate 1.8016905856883567Mbps', 'rate 7.420240584848075Mbps', 'rate 7.783550714605846Mbps', 'rate 8.009219812303668Mbps', 'rate 9.866335465046115Mbps', 'rate 8.937930782250625Mbps', 'rate 6.521726232464559Mbps', 'rate 8.913587305195481Mbps', 'rate 1.7777017572672908Mbps', 'rate 1.7658662889593597Mbps', 'rate 1.4955549752266242Mbps', 'rate 8.325799818666296Mbps', 'rate 1.5118504229153018Mbps', 'rate 8.714937657406026Mbps', 'rate 1.4494967820600566Mbps', 'rate 1.2261783625967868Mbps', 'rate 1.9826773702098754Mbps', 'rate 7.309987516368306Mbps', 'rate 8.087845995386676Mbps', 'rate 1.099555764700344Mbps', 'rate 8.942441217576011Mbps', 'rate 5.883725085027172Mbps', 'rate 9.19485560853276Mbps', 'rate 4.042404692938062Mbps', 'rate 9.176054136865279Mbps', 'rate 9.242678476721592Mbps', 'rate 7.30840193725335Mbps', 'rate 1.8629485945401227Mbps', 'rate 9.569957170874973Mbps', 'rate 9.42831638159816Mbps', 'rate 5.340122129000905Mbps', 'rate 1.0122785124092053Mbps', 'rate 5.925157611840351Mbps', 'rate 4.615336620291671Mbps', 'rate 1.744956589933429Mbps', 'rate 5.607266732004014Mbps', 'rate 1.1730826480050731Mbps', 'rate 9.069956763096451Mbps', 'rate 8.151243741085533Mbps', 'rate 9.840474146531442Mbps', 'rate 4.678762706648028Mbps', 'rate 1.5599967971107347Mbps', 'rate 9.573743030033391Mbps', 'rate 8.585629340462738Mbps', 'rate 6.8030106498331655Mbps', 'rate 9.430191382986385Mbps', 'rate 5.9456159390959735Mbps', 'rate 8.23221244892307Mbps', 'rate 6.575831781483702Mbps', 'rate 1.7911536445503524Mbps', 'rate 1.0063958521228078Mbps', 'rate 8.878672491869525Mbps', 'rate 9.112768919512996Mbps', 'rate 1.5373027232713785Mbps', 'rate 6.483300790363366Mbps', 'rate 1.6845399931493859Mbps', 'rate 1.544196704429294Mbps', 'rate 5.699348672148009Mbps', 'rate 4.100305552514886Mbps', 'rate 1.112766320673832Mbps', 'rate 1.0896243716118175Mbps', 'rate 5.015080363477345Mbps', 'rate 7.2394748490171565Mbps', 'rate 5.02747247748011Mbps', 'rate 8.454217230562078Mbps', 'rate 1.1722110362398435Mbps', 'rate 8.54471980962542Mbps', 'rate 6.473360812593674Mbps', 'rate 1.8766347085407222Mbps', 'rate 8.561515041786581Mbps', 'rate 5.447962006137571Mbps', 'rate 1.03656490232216Mbps', 'rate 8.446267007408197Mbps', 'rate 9.790787625652536Mbps', 'rate 8.772227888736772Mbps', 'rate 6.570791881815426Mbps', 'rate 6.336013352637083Mbps', 'rate 6.657365472716634Mbps', 'rate 1.7204365230929728Mbps', 'rate 1.3658821668128234Mbps', 'rate 1.3982812386813532Mbps', 'rate 1.9136932030032257Mbps', 'rate 4.149052777962242Mbps', 'rate 4.018796328272021Mbps', 'rate 1.393441067085376Mbps', 'rate 9.855676904433402Mbps', 'rate 1.3588703954684127Mbps', 'rate 8.115958667993365Mbps', 'rate 8.80926647488175Mbps', 'rate 1.0482834271764911Mbps', 'rate 9.266037235701168Mbps', 'rate 5.6560178998439135Mbps', 'rate 1.4756802525511206Mbps', 'rate 7.661556749361502Mbps', 'rate 1.7760338767879218Mbps', 'rate 1.6690210136499108Mbps', 'rate 9.350965022893096Mbps', 'rate 8.042056658942377Mbps', 'rate 7.834942117135419Mbps', 'rate 1.1492684023547275Mbps', 'rate 7.018729052705265Mbps', 'rate 1.5339230270092923Mbps', 'rate 6.563318384068494Mbps', 'rate 1.1501953316505429Mbps', 'rate 5.641122388760294Mbps', 'rate 9.779191116216236Mbps', 'rate 4.642995292248686Mbps', 'rate 1.9111810620251868Mbps', 'rate 1.498645677569324Mbps', 'rate 1.0835234768415751Mbps', 'rate 8.18589407183389Mbps', 'rate 1.743674799571413Mbps', 'rate 5.25432454074482Mbps', 'rate 8.106344191516788Mbps', 'rate 1.3044224067815235Mbps', 'rate 5.213766378050104Mbps', 'rate 8.559677874334813Mbps', 'rate 8.479252823723357Mbps', 'rate 6.45725191689416Mbps', 'rate 1.5902821847779465Mbps', 'rate 8.790762895816245Mbps', 'rate 1.7155563653396158Mbps', 'rate 4.368792689628262Mbps', 'rate 9.50260452291464Mbps', 'rate 8.92042383230191Mbps', 'rate 1.8017066616293769Mbps', 'rate 9.200837762079011Mbps', 'rate 5.627987007544468Mbps', 'rate 5.4962130241005775Mbps', 'rate 1.4837631247607002Mbps', 'rate 6.604060108264529Mbps', 'rate 1.7832741748650198Mbps', 'rate 5.406359916327174Mbps', 'rate 4.948240913713779Mbps', 'rate 1.0921838639765786Mbps', 'rate 9.771878028188292Mbps', 'rate 7.478024829699937Mbps', 'rate 1.6835102482846558Mbps', 'rate 9.79194246341391Mbps', 'rate 8.580879976307354Mbps', 'rate 8.707234396793256Mbps', 'rate 6.496002339269756Mbps', 'rate 9.770346841060203Mbps', 'rate 9.573947570240808Mbps', 'rate 4.415053595342586Mbps', 'rate 6.322971762833677Mbps', 'rate 9.16161686929337Mbps', 'rate 8.842123284765607Mbps', 'rate 9.334320215351948Mbps', 'rate 1.1583672935821658Mbps', 'rate 6.564152196388105Mbps', 'rate 6.456341082985796Mbps', 'rate 1.591586799853814Mbps', 'rate 9.497691357122372Mbps', 'rate 1.8056847596774612Mbps', 'rate 4.111709493097128Mbps', 'rate 9.67489939457527Mbps', 'rate 9.361497653576855Mbps', 'rate 4.479491102909787Mbps', 'rate 9.387587015382493Mbps', 'rate 9.402670857795773Mbps', 'rate 6.601510459242339Mbps', 'rate 7.52224783369114Mbps', 'rate 7.444820402108629Mbps', 'rate 9.258670906949693Mbps', 'rate 9.848386396845857Mbps', 'rate 9.496675434492818Mbps', 'rate 5.700299676790766Mbps', 'rate 4.898653143124504Mbps', 'rate 7.235825671643842Mbps', 'rate 4.481564436466497Mbps', 'rate 8.331971434175628Mbps', 'rate 1.9866626771869305Mbps', 'rate 9.666350355679587Mbps', 'rate 1.9438800307452349Mbps', 'rate 1.5303622195231563Mbps', 'rate 1.7254944080627705Mbps', 'rate 1.5295314756967033Mbps', 'rate 9.835854231760965Mbps', 'rate 1.9257700537203064Mbps', 'rate 8.582522726079416Mbps', 'rate 8.317339191642498Mbps', 'rate 1.187601855312784Mbps', 'rate 1.5784448812206473Mbps', 'rate 1.304504013248386Mbps', 'rate 9.373305632583874Mbps', 'rate 8.50146617427101Mbps', 'rate 1.3371314782397699Mbps', 'rate 1.613471250311303Mbps', 'rate 4.696349456358579Mbps', 'rate 6.326888745333578Mbps', 'rate 1.258079834409233Mbps', 'rate 9.700179583009334Mbps', 'rate 1.1154382276737125Mbps', 'rate 9.735623807013592Mbps', 'rate 4.373330240090537Mbps', 'rate 8.424969882027533Mbps', 'rate 7.18450447464627Mbps', 'rate 4.677308636281007Mbps', 'rate 8.849045595806498Mbps', 'rate 4.9878648750933685Mbps', 'rate 4.952720890959069Mbps', 'rate 8.6844555079027Mbps', 'rate 1.6368281973984975Mbps', 'rate 4.053764184092811Mbps', 'rate 9.052308047814984Mbps', 'rate 6.808512742883545Mbps', 'rate 1.8996297802610647Mbps', 'rate 8.702493534803374Mbps', 'rate 4.424235641640311Mbps', 'rate 9.491744774936466Mbps', 'rate 7.177002957332098Mbps', 'rate 4.682933848513846Mbps', 'rate 8.948404757952106Mbps', 'rate 6.9782792912604865Mbps', 'rate 1.1201412057514688Mbps', 'rate 9.405446625924883Mbps', 'rate 1.4141127243382745Mbps', 'rate 1.8392330083984885Mbps', 'rate 7.009903978127537Mbps', 'rate 1.8212115474134016Mbps', 'rate 1.316032295405173Mbps', 'rate 1.4533096890830723Mbps', 'rate 1.4085261881933984Mbps', 'rate 4.574527848507216Mbps', 'rate 9.656533304823686Mbps', 'rate 6.227460471071092Mbps', 'rate 1.9270460004527363Mbps', 'rate 1.4630235804853795Mbps', 'rate 8.450525497133953Mbps', 'rate 8.518960789443277Mbps', 'rate 9.03155887658059Mbps', 'rate 8.87512844442704Mbps', 'rate 8.16032353960985Mbps', 'rate 9.023724784980088Mbps', 'rate 1.7808595156128129Mbps', 'rate 6.80958996859761Mbps', 'rate 8.785875735077424Mbps', 'rate 1.5363172539389187Mbps', 'rate 8.654663495637081Mbps', 'rate 1.0687601134535876Mbps', 'rate 9.340372483252885Mbps', 'rate 6.178432490075335Mbps', 'rate 9.274990585313928Mbps', 'rate 1.217410177487733Mbps', 'rate 6.141994262825936Mbps', 'rate 8.217570026128985Mbps', 'rate 1.6272180705258887Mbps', 'rate 9.602468358051365Mbps', 'rate 9.189693904019775Mbps', 'rate 1.86525555369108Mbps', 'rate 6.546992813985604Mbps', 'rate 8.027344315567968Mbps', 'rate 8.69194745132191Mbps', 'rate 1.5326486576783123Mbps', 'rate 9.566701161739552Mbps', 'rate 8.675351608140806Mbps', 'rate 5.318403530380239Mbps', 'rate 9.761531237906642Mbps', 'rate 9.068409571240046Mbps', 'rate 5.829661613441564Mbps', 'rate 8.956873720784442Mbps', 'rate 9.016514114657276Mbps', 'rate 8.042823948617418Mbps', 'rate 8.360275175642851Mbps', 'rate 8.517874993949755Mbps', 'rate 4.791433893411192Mbps', 'rate 1.6492790230733472Mbps', 'rate 9.549294595172261Mbps', 'rate 4.878148657680843Mbps', 'rate 5.767698935742024Mbps', 'rate 1.9764317247939651Mbps', 'rate 9.319219764475875Mbps', 'rate 1.0004987505363128Mbps', 'rate 1.9966469774483373Mbps', 'rate 1.141895769796139Mbps', 'rate 7.844021279833922Mbps', 'rate 1.1587962465159931Mbps', 'rate 8.443015491238313Mbps', 'rate 1.9007673368744311Mbps', 'rate 8.481114192937032Mbps', 'rate 8.374563210161446Mbps', 'rate 1.249220156697663Mbps', 'rate 9.330682329098217Mbps']
        
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
