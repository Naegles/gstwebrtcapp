from abc import ABCMeta, abstractmethod
import collections
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, OrderedDict, Tuple

from control.drl.reward import RewardFunctionFactory
from utils.base import LOGGER, scale, unscale, merge_observations
from utils.gst import GstWebRTCStatsType, find_stat, get_stat_diff, get_stat_diff_concat
from utils.webrtc import clock_units_to_seconds, ntp_short_format_to_seconds


class MDP(metaclass=ABCMeta):
    '''
    MDP is an abstract class for Markov Decision Process. It defines the interface for the environment.
    It also provides methods to translate the environment state and action to the controller's state and action
    that could be directly applied to the GStreamer pipeline and vice versa.
    '''

    MAX_BITRATE_STREAM_MBPS = 10.0  # so far for 1 stream only, later we can have multiple streams
    MIN_BITRATE_STREAM_MBPS = 0.4  # so far for 1 stream only, later we can have multiple streams
    MAX_BANDWIDTH_MBPS = 20.0
    MIN_BANDWIDTH_MBPS = 0.4
    MAX_DELAY_SEC = 1  # assume we target the sub-second latency

    CONSTANTS = {
        "MAX_BITRATE_STREAM_MBPS": MAX_BITRATE_STREAM_MBPS,
        "MIN_BITRATE_STREAM_MBPS": MIN_BITRATE_STREAM_MBPS,
        "MAX_BANDWIDTH_MBPS": MAX_BANDWIDTH_MBPS,
        "MIN_BANDWIDTH_MBPS": MIN_BANDWIDTH_MBPS,
        "MAX_DELAY_SEC": MAX_DELAY_SEC,
    }

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 1,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.reward_function = RewardFunctionFactory().create_reward_function(reward_function_name)
        self.episode_length = episode_length
        self.num_observations_for_state = num_observations_for_state
        self.state_history_size = state_history_size
        if constants is not None:
            for key in constants:
                self.CONSTANTS[key] = constants[key]

        self.mqtts = None
        self.states_made = 0
        self.is_scaled = False
        self.last_stats = None
        self.last_states = collections.deque(maxlen=self.state_history_size + 1)
        self.max_rb_packetslost = 0
        self.first_ssrc = None
        self.obs_filter = None

    @abstractmethod
    def reset(self):
        # memento pattern
        self.first_ssrc = None
        self.states_made = 0
        self.last_stats = None
        self.last_states = collections.deque(maxlen=self.state_history_size + 1)

    @abstractmethod
    def create_observation_space(self) -> spaces.Dict:
        pass

    @abstractmethod
    def create_action_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def make_default_state(self) -> OrderedDict[str, Any]:
        pass

    @abstractmethod
    def make_state(self, stats: Dict[str, Any]) -> OrderedDict[str, Any]:
        self.states_made += 1
        pass

    @abstractmethod
    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        pass

    @abstractmethod
    def convert_to_unscaled_action(self, action: Any) -> Any:
        pass

    @abstractmethod
    def check_observation(self, obs: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        pass

    def calculate_reward(self) -> Tuple[float, Dict[str, Any | float] | None]:
        return self.reward_function.calculate_reward(self.last_states)

    def get_default_reward_parts_dict(self) -> Dict[str, Any | float] | None:
        return dict(zip(self.reward_function.reward_parts, [0.0] * len(self.reward_function.reward_parts)))

    def check_observation(self, obs: Dict[str, Any]) -> bool:
        # rtp inbound stream is the most important stat
        rtp_inbounds = find_stat(obs, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        if not rtp_inbounds:
            return False

        # filter over other stats
        if self.obs_filter is not None:
            for stat in self.obs_filter:
                if stat != GstWebRTCStatsType.RTP_INBOUND_STREAM:
                    if not find_stat(obs, stat):
                        return False

        # check that pl not smaller than last max seen packet lost
        for rtp_inbound in rtp_inbounds:
            # haven't seen any packet lost yet
            if self.first_ssrc is None:
                return True
            else:
                if rtp_inbound["ssrc"] == self.first_ssrc:
                    rb_packetslost = rtp_inbound["rb-packetslost"]
                    # assumed that packet lost increases more or less in the same manner
                    if rb_packetslost < self.max_rb_packetslost:
                        return False
                    else:
                        self.max_rb_packetslost = rb_packetslost
                        return True

    def is_terminated(self, step: int) -> bool:
        return False

    def is_truncated(self, step) -> bool:
        return step >= self.episode_length


class ViewerMDP(MDP):
    '''
    This MDP takes VIEWER (aka BROWSER) stats delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 1,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            state_history_size,
            constants,
        )

        # obs are scaled to [0, 1], actions are scaled to [-1, 1]
        self.is_scaled = True

        # filter obs only with needed stats in
        self.obs_filter = [
            GstWebRTCStatsType.RTP_OUTBOUND_STREAM,
            GstWebRTCStatsType.RTP_INBOUND_STREAM,
            GstWebRTCStatsType.ICE_CANDIDATE_PAIR,
        ]

        self.reset()

    def reset(self):
        super().reset()
        self.rtts = []

    def create_observation_space(self) -> spaces.Dict:
        # normalized to [0, 1]
        return spaces.Dict(
            {
                "bandwidth": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "fractionLossRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionNackRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionPliRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionQueueingRtt": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionRtt": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "interarrivalRttJitter": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "lossRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttMean": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttStd": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rxGoodput": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "txGoodput": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

    def create_action_space(self) -> spaces.Space:
        # basic AS uses only bitrate, normalized to [-1, 1]
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def make_default_state(self) -> OrderedDict[str, Any]:
        return collections.OrderedDict(
            {
                "bandwidth": [0.0, 0.0],
                "fractionLossRate": 0.0,
                "fractionNackRate": 0.0,
                "fractionPliRate": 0.0,
                "fractionQueueingRtt": 0.0,
                "fractionRtt": 0.0,
                "interarrivalRttJitter": 0.0,
                "lossRate": 0.0,
                "rttMean": 0.0,
                "rttStd": 0.0,
                "rxGoodput": 0.0,
                "txGoodput": 0.0,
            }
        )

    def make_state(self, stats: Dict[str, Any]) -> OrderedDict[str, Any]:
        super().make_state(stats)
        # get gcc bandiwdth
        bws = []
        while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.gcc].empty():
            msg = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.gcc)
            bws.append(float(msg.msg))
        bws = [b / 1e6 for b in bws]
        if len(bws) == 0:
            if not self.last_states:
                bandwidth = [0.0, 0.0]
            else:
                bandwidth = [1.0, 1.0]
        elif len(bws) == 1:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]
        else:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[-1], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]
        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")
                ts_diff_sec = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp") / 1000

                # loss rates
                # 1. fraction loss rate
                rb_packetslost_diff = get_stat_diff(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rate = (
                    rb_packetslost_diff / (packets_sent_diff + rb_packetslost_diff)
                    if packets_sent_diff + rb_packetslost_diff > 0
                    else 0
                )
                fraction_loss_rate = max(0, min(1, fraction_loss_rate))
                # 7. global loss rate
                loss_rate = (
                    rtp_inbound[i]["rb-packetslost"]
                    / (rtp_outbound[0]["packets-sent"] + rtp_inbound[i]["rb-packetslost"])
                    if rtp_outbound[0]["packets-sent"] + rtp_inbound[i]["rb-packetslost"] > 0
                    else 0.0
                )

                # 2. fraction nack rate
                recv_nack_count_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rate = recv_nack_count_diff / packets_recv_diff if packets_recv_diff > 0 else 0.0

                # 3. fraction pli rate
                recv_pli_count_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rate = recv_pli_count_diff / packets_recv_diff if packets_recv_diff > 0 else 0.0

                # rtts: RTT comes in NTP short format
                rtt = ntp_short_format_to_seconds(rtp_inbound[i]["rb-round-trip"]) / self.CONSTANTS["MAX_DELAY_SEC"]
                self.rtts.append(rtt)

                # 4. fraction queueing rtt
                fraction_queueing_rtt = rtt - min(self.rtts) if len(self.rtts) > 0 else 0.0
                # 9. mean rtt
                rtt_mean = np.mean(self.rtts) if len(self.rtts) > 0 else 0.0
                # 10. std rtt
                rtt_std = np.std(self.rtts) if len(self.rtts) > 0 else 0.0

                # 6. jitter: comes in clock units
                interarrival_jitter = (
                    clock_units_to_seconds(rtp_inbound[i]["rb-jitter"], rtp_outbound[0]["clock-rate"])
                    / self.CONSTANTS["MAX_DELAY_SEC"]
                )

                # 11. rx rate
                try:
                    bitrate_recv = ice_candidate_pair[0]["bitrate-recv"]
                    rx_rate = scale(
                        bitrate_recv / 1000000,
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    )
                except KeyError:
                    rx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                    rx_mbits_diff = rx_bytes_diff * 8 / 1000000
                    rx_rate = rx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0
                    rx_rate = scale(
                        rx_rate, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"]
                    )

                # 12. tx rate
                try:
                    bitrate_sent = ice_candidate_pair[0]["bitrate-sent"]
                    tx_rate = scale(
                        bitrate_sent / 1000000,
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    )
                except KeyError:
                    tx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-sent")
                    tx_mbits_diff = tx_bytes_diff * 8 / 1000000
                    tx_rate = tx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0
                    tx_rate = scale(
                        tx_rate, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"]
                    )

                # form the final state
                state = collections.OrderedDict(
                    {
                        "bandwidth": bandwidth,
                        "fractionLossRate": fraction_loss_rate,
                        "fractionNackRate": fraction_nack_rate,
                        "fractionPliRate": fraction_pli_rate,
                        "fractionQueueingRtt": fraction_queueing_rtt,
                        "fractionRtt": rtt,
                        "interarrivalRttJitter": interarrival_jitter,
                        "lossRate": loss_rate,
                        "rttMean": rtt_mean,
                        "rttStd": rtt_std,
                        "rxGoodput": rx_rate,
                        "txGoodput": tx_rate,
                    }
                )

                self.last_states.append(state)
                return state

        LOGGER.warning("WARNING: Drl Agent: ViewerMDP: make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return (
            collections.OrderedDict(
                {
                    "bandwidth": [
                        unscale(s, self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"])
                        for s in state["bandwidth"]
                    ],
                    "fractionLossRate": state["fractionLossRate"],
                    "fractionNackRate": state["fractionNackRate"],
                    "fractionPliRate": state["fractionPliRate"],
                    "fractionQueueingRtt": state["fractionQueueingRtt"] * self.MAX_DELAY_SEC,
                    "fractionRtt": state["fractionRtt"] * self.MAX_DELAY_SEC,
                    "interarrivalRttJitter": state["interarrivalRttJitter"] * self.MAX_DELAY_SEC,
                    "lossRate": state["lossRate"],
                    "rttMean": state["rttMean"] * self.MAX_DELAY_SEC,
                    "rttStd": state["rttStd"] * self.MAX_DELAY_SEC,
                    "rxGoodput": unscale(
                        state["rxGoodput"],
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    ),
                    "txGoodput": unscale(
                        state["txGoodput"],
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    ),
                }
            )
            if self.is_scaled
            else state
        )

    def convert_to_unscaled_action(self, action: np.ndarray | float | int) -> np.ndarray | float:
        return self.MIN_BITRATE_STREAM_MBPS + (
            (action + 1) * (self.MAX_BITRATE_STREAM_MBPS - self.MIN_BITRATE_STREAM_MBPS) / 2
        )

    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        # here we have only bitrate decisions that come as a 1-size np array in mbps (check create_action_space)
        return {"bitrate": self.convert_to_unscaled_action(action)[0] * 1000}


class ViewerSeqMDP(MDP):
    '''
    This MDP takes VIEWER (aka BROWSER) sequential stats (stacked observations) delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 5,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            state_history_size,
            constants,
        )

        # obs are scaled to [0, 1], actions are scaled to [-1, 1]
        self.is_scaled = True

        # filter obs only with needed stats in
        self.obs_filter = [
            GstWebRTCStatsType.RTP_OUTBOUND_STREAM,
            GstWebRTCStatsType.RTP_INBOUND_STREAM,
            GstWebRTCStatsType.ICE_CANDIDATE_PAIR,
        ]

        self.reset()

    def reset(self):
        super().reset()
        self.rtts = []

    def create_observation_space(self) -> spaces.Dict:
        # normalized to [0, 1]
        shape = (self.num_observations_for_state,)
        return spaces.Dict(
            {
                "bandwidth": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "fractionLossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionNackRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionPliRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionQueueingRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "interarrivalRttJitter": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "lossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "rttMean": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttStd": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rxGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "txGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
            }
        )

    def create_action_space(self) -> spaces.Space:
        # basic AS uses only bitrate, normalized to [-1, 1]
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def make_default_state(self) -> OrderedDict[str, Any]:
        def_val = 0.0 if self.num_observations_for_state == 1 else [0.0] * self.num_observations_for_state
        return collections.OrderedDict(
            {
                "bandwidth": [0.0, 0.0],
                "fractionLossRate": def_val,
                "fractionNackRate": def_val,
                "fractionPliRate": def_val,
                "fractionQueueingRtt": def_val,
                "fractionRtt": def_val,
                "interarrivalRttJitter": def_val,
                "lossRate": def_val,
                "rttMean": 0.0,
                "rttStd": 0.0,
                "rxGoodput": def_val,
                "txGoodput": def_val,
            }
        )

    def make_state(self, stats: Dict[str, Any]) -> OrderedDict[str, Any]:
        super().make_state(stats)
        # get gcc bandiwdth
        bws = []
        while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.gcc].empty():
            msg = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.gcc)
            bws.append(float(msg.msg))
        bws = [b / 1e6 for b in bws]
        if len(bws) == 0:
            if not self.last_states:
                bandwidth = [0.0, 0.0]
            else:
                bandwidth = [1.0, 1.0]
        elif len(bws) == 1:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]
        else:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[-1], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]
        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"][0]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"][0] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")
                ts_diff_sec = [
                    ts / 1000 for ts in get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp")
                ]

                # loss rates
                # 1. fraction loss rate
                rb_packetslost_diff = get_stat_diff_concat(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rates = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rb_packetslost_diff, packets_sent_diff)
                ]
                # 7. global loss rate
                loss_rates = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rtp_inbound[i]["rb-packetslost"], rtp_outbound[0]["packets-sent"])
                ]

                # 2. fraction nack rate
                recv_nack_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rates = [
                    nack / recv if recv > 0 else 0.0 for nack, recv in zip(recv_nack_count_diff, packets_recv_diff)
                ]

                # 3. fraction pli rate
                recv_pli_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rates = [
                    pli / recv if recv > 0 else 0.0 for pli, recv in zip(recv_pli_count_diff, packets_recv_diff)
                ]

                # rtts: RTT comes in NTP short format
                rtts = [
                    ntp_short_format_to_seconds(rtt) / self.CONSTANTS["MAX_DELAY_SEC"]
                    for rtt in rtp_inbound[i]["rb-round-trip"]
                ]
                self.rtts.extend(rtts)

                # 4. fraction queueing rtt
                fraction_queueing_rtts = [rtt - min(self.rtts) if len(self.rtts) > 0 else 0.0 for rtt in rtts]
                # 9. mean rtt
                rtt_mean = np.mean(self.rtts) if len(self.rtts) > 0 else 0.0
                # 10. std rtt
                rtt_std = np.std(self.rtts) if len(self.rtts) > 0 else 0.0

                # 6. jitter: comes in clock units
                interarrival_jitters = [
                    clock_units_to_seconds(j, rtp_outbound[0]["clock-rate"][0]) / self.CONSTANTS["MAX_DELAY_SEC"]
                    for j in rtp_inbound[i]["rb-jitter"]
                ]

                # 11. rx rate
                try:
                    bitrate_recvs = ice_candidate_pair[0]["bitrate-recv"]
                    rx_rates = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_recvs
                    ]
                except KeyError:
                    rx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                    rx_mbits_diff = [r * 8 / 1000000 for r in rx_bytes_diff]
                    rx_rates = [r / ts if ts > 0 else 0.0 for r, ts in zip(rx_mbits_diff, ts_diff_sec)]
                    rx_rates = [
                        scale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in rx_rates
                    ]

                # 12. tx rate
                try:
                    bitrate_sents = ice_candidate_pair[0]["bitrate-sent"]
                    tx_rates = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_sents
                    ]
                except KeyError:
                    tx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-sent")
                    tx_mbits_diff = [t * 8 / 1000000 for t in tx_bytes_diff]
                    tx_rates = [t / ts if ts > 0 else 0.0 for t, ts in zip(tx_mbits_diff, ts_diff_sec)]
                    tx_rates = [
                        scale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in tx_rates
                    ]

                # form the final state
                state = collections.OrderedDict(
                    {
                        "bandwidth": bandwidth,
                        "fractionLossRate": fraction_loss_rates,
                        "fractionNackRate": fraction_nack_rates,
                        "fractionPliRate": fraction_pli_rates,
                        "fractionQueueingRtt": fraction_queueing_rtts,
                        "fractionRtt": rtts,
                        "interarrivalRttJitter": interarrival_jitters,
                        "lossRate": loss_rates,
                        "rttMean": rtt_mean,
                        "rttStd": rtt_std,
                        "rxGoodput": rx_rates,
                        "txGoodput": tx_rates,
                    }
                )

                self.last_states.append(state)
                return state

        LOGGER.warning("WARNING: Drl Agent: ViewerMDP: make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return (
            collections.OrderedDict(
                {
                    "bandwidth": [
                        unscale(s, self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"])
                        for s in state["bandwidth"]
                    ],
                    "fractionLossRate": state["fractionLossRate"],
                    "fractionNackRate": state["fractionNackRate"],
                    "fractionPliRate": state["fractionPliRate"],
                    "fractionQueueingRtt": [fqr * self.MAX_DELAY_SEC for fqr in state["fractionQueueingRtt"]],
                    "fractionRtt": [fr * self.MAX_DELAY_SEC for fr in state["fractionRtt"]],
                    "interarrivalRttJitter": [irj * self.MAX_DELAY_SEC for irj in state["interarrivalRttJitter"]],
                    "lossRate": state["lossRate"],
                    "rttMean": state["rttMean"] * self.MAX_DELAY_SEC,
                    "rttStd": state["rttStd"] * self.MAX_DELAY_SEC,
                    "rxGoodput": [
                        unscale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in state["rxGoodput"]
                    ],
                    "txGoodput": [
                        unscale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in state["txGoodput"]
                    ],
                }
            )
            if self.is_scaled
            else state
        )

    def convert_to_unscaled_action(self, action: np.ndarray | float | int) -> np.ndarray | float:
        return self.MIN_BITRATE_STREAM_MBPS + (
            (action + 1) * (self.MAX_BITRATE_STREAM_MBPS - self.MIN_BITRATE_STREAM_MBPS) / 2
        )

    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        # here we have only bitrate decisions that come as a 1-size np array in mbps (check create_action_space)
        return {"bitrate": self.convert_to_unscaled_action(action)[0] * 1000}
