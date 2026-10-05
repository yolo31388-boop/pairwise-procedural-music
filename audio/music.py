"""
游戏内程序化音乐生成与音频系统 - 核心模块

功能：
- generate_melody(): 基于音阶与节奏的旋律生成
- audio_synthesis(): 振荡器（正弦/方波/锯齿波）+ ADSR 包络的音频合成
- music_theory(): 音阶、和弦进行与调式
- adaptive_music(): 根据游戏场景切换情绪并平滑过渡
- sound_design(): 带距离层次与随机变化的音效设计
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


NOTE_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MODES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}

# 和弦进行（音阶级数序列）
CHORD_PROGRESSIONS = {
    "major": [0, 4, 5, 3],   # I-V-vi-IV
    "minor": [0, 5, 2, 6],   # i-VI-III-VII
    "dorian": [0, 3, 0, 4],
    "mixolydian": [0, 6, 3, 0],
}

SCENE_EMOTIONS = {
    "menu": {"mood": "calm", "tempo": 70.0, "mode": "major", "intensity": 0.2},
    "exploration": {"mood": "calm", "tempo": 90.0, "mode": "major", "intensity": 0.4},
    "combat": {"mood": "intense", "tempo": 140.0, "mode": "minor", "intensity": 0.9},
    "boss": {"mood": "epic", "tempo": 160.0, "mode": "minor", "intensity": 1.0},
}

SOUND_EVENTS = {
    "footstep": {"waveform": "sine", "frequency": 120.0, "duration": 0.15},
    "sword_hit": {"waveform": "square", "frequency": 800.0, "duration": 0.3},
    "explosion": {"waveform": "sawtooth", "frequency": 60.0, "duration": 1.2},
    "ui_click": {"waveform": "sine", "frequency": 1000.0, "duration": 0.05},
}

RHYTHM_CHOICES = [0.25, 0.5, 0.5, 0.75, 1.0, 1.5]


@dataclass
class Config:
    pass


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


class ProceduralMusic:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._rng = random.Random()

    # ---------------- 旋律生成 ----------------

    def generate_melody(self, length: int = 8, key: str = "C",
                        mode: str = "major", octave: int = 4,
                        seed: Optional[int] = None) -> List[Dict]:
        """基于音阶随机游走生成旋律，每个音符带有节奏时长。"""
        if mode not in MODES:
            raise ValueError(f"未知调式: {mode}")
        if length <= 0:
            return []
        rng = random.Random(seed)
        root_midi = 12 * (octave + 1) + NOTE_OFFSETS[key]
        intervals = MODES[mode]
        scale_midi = [root_midi + iv for iv in intervals]

        melody = []
        degree = 0
        for _ in range(length):
            midi = scale_midi[degree]
            duration = rng.choice(RHYTHM_CHOICES)
            melody.append({
                "note": NOTE_NAMES[midi % 12],
                "midi": midi,
                "frequency": _midi_to_freq(midi),
                "duration": duration,
            })
            degree = max(0, min(len(scale_midi) - 1, degree + rng.randint(-2, 2)))
        self._history.append(("melody", length, key, mode))
        return melody

    # ---------------- 音频合成 ----------------

    def audio_synthesis(self, waveform: str = "sine", frequency: float = 440.0,
                        duration: float = 1.0, sample_rate: int = 8000,
                        envelope: Tuple[float, float, float, float] = (0.1, 0.1, 0.7, 0.2)
                        ) -> List[float]:
        """用振荡器生成波形并施加 ADSR 包络，返回采样点列表。

        envelope = (attack, decay, sustain_level, release)，
        其中 attack/decay/release 为占时长的比例，sustain_level 为电平。
        """
        oscillators = {
            "sine": lambda t: math.sin(2.0 * math.pi * frequency * t),
            "square": lambda t: 1.0 if math.sin(2.0 * math.pi * frequency * t) >= 0 else -1.0,
            "sawtooth": lambda t: 2.0 * ((frequency * t) % 1.0) - 1.0,
            "triangle": lambda t: 2.0 * abs(2.0 * ((frequency * t) % 1.0) - 1.0) - 1.0,
        }
        if waveform not in oscillators:
            raise ValueError(f"未知波形: {waveform}")
        osc = oscillators[waveform]
        attack, decay, sustain, release = envelope

        def env(pos: float) -> float:
            if pos < attack:
                return pos / attack if attack > 0 else 1.0
            if pos < attack + decay:
                return 1.0 - (1.0 - sustain) * ((pos - attack) / decay) if decay > 0 else sustain
            if pos < 1.0 - release:
                return sustain
            if release > 0:
                return sustain * (1.0 - (pos - (1.0 - release)) / release)
            return sustain

        count = max(1, int(duration * sample_rate))
        samples = []
        for i in range(count):
            t = i / sample_rate
            pos = i / count
            samples.append(osc(t) * env(pos))
        return samples

    # ---------------- 音乐理论 ----------------

    def music_theory(self, key: str = "C", mode: str = "major") -> Dict:
        """返回指定调性/调式下的音阶、各级三和弦与和弦进行。"""
        if mode not in MODES:
            raise ValueError(f"未知调式: {mode}")
        root = NOTE_OFFSETS[key]
        intervals = MODES[mode]
        scale = [NOTE_NAMES[(root + iv) % 12] for iv in intervals]
        n = len(intervals)

        def triad(degree: int) -> List[str]:
            notes = []
            for step in (0, 2, 4):
                idx = degree + step
                semitone = root + intervals[idx % n] + 12 * (idx // n)
                notes.append(NOTE_NAMES[semitone % 12])
            return notes

        def triad_quality(degree: int) -> str:
            third = (intervals[(degree + 2) % n] + 12 * ((degree + 2) // n)
                     - intervals[degree % n] - 12 * (degree // n)) % 12
            fifth = (intervals[(degree + 4) % n] + 12 * ((degree + 4) // n)
                     - intervals[degree % n] - 12 * (degree // n)) % 12
            if third == 4 and fifth == 7:
                return "major"
            if third == 3 and fifth == 7:
                return "minor"
            return "diminished"

        numerals = ["I", "II", "III", "IV", "V", "VI", "VII"]
        chords = {}
        for degree in range(n):
            quality = triad_quality(degree)
            numeral = numerals[degree] if quality == "major" else numerals[degree].lower()
            chords[numeral] = triad(degree)

        progression_degrees = CHORD_PROGRESSIONS.get(mode, CHORD_PROGRESSIONS["major"])
        progression = [triad(d % n) for d in progression_degrees]

        return {
            "key": key,
            "mode": mode,
            "is_major": mode in ("major", "lydian", "mixolydian"),
            "intervals": list(intervals),
            "scale": scale,
            "chords": chords,
            "chord_progression": progression,
        }

    # ---------------- 自适应音乐 ----------------

    def adaptive_music(self, scene: str, transition_time: float = 2.0) -> Dict:
        """根据场景切换音乐情绪，切换时按 transition_time 秒平滑过渡。"""
        if scene not in SCENE_EMOTIONS:
            raise ValueError(f"未知场景: {scene}")
        target = dict(SCENE_EMOTIONS[scene])
        current_scene = self._state.get("scene")

        if current_scene is None:
            self._state["scene"] = scene
            self._state["emotion"] = target
            self._state["transition"] = None
        elif current_scene != scene:
            self._state["transition"] = {
                "from": dict(self._state["emotion"]),
                "to": target,
                "elapsed": 0.0,
                "duration": max(transition_time, 1e-6),
            }
            self._state["scene"] = scene
        return self._adaptive_state()

    def _adaptive_state(self) -> Dict:
        transition = self._state.get("transition")
        if transition:
            progress = min(1.0, transition["elapsed"] / transition["duration"])
            emotion = self._blend_emotions(transition["from"], transition["to"], progress)
        else:
            progress = 1.0
            emotion = dict(self._state.get("emotion", {}))
        return {
            "scene": self._state.get("scene"),
            "emotion": emotion,
            "transitioning": transition is not None,
            "progress": progress,
        }

    @staticmethod
    def _blend_emotions(src: Dict, dst: Dict, t: float) -> Dict:
        blended = {}
        for key, value in dst.items():
            if isinstance(value, (int, float)) and isinstance(src.get(key), (int, float)):
                blended[key] = src[key] + (value - src[key]) * t
            else:
                blended[key] = value if t >= 0.5 else src.get(key, value)
        return blended

    # ---------------- 音效设计 ----------------

    def sound_design(self, event: str, distance: float = 0.0,
                     seed: Optional[int] = None) -> Dict:
        """生成带距离层次与随机变化的音效描述。

        距离越远音量越衰减，并叠加反射/混响层；
        同一事件每次触发的频率与时长都有随机变化（可用 seed 固定）。
        """
        if event not in SOUND_EVENTS:
            raise ValueError(f"未知音效事件: {event}")
        rng = random.Random(seed) if seed is not None else self._rng
        base = SOUND_EVENTS[event]
        distance = max(0.0, distance)

        volume = 1.0 / (1.0 + distance * 0.1)
        frequency = base["frequency"] * rng.uniform(0.95, 1.05)
        duration = base["duration"] * rng.uniform(0.9, 1.1)

        layers = [{"name": "dry", "volume": volume}]
        if distance > 5.0:
            layers.append({"name": "reflection", "volume": volume * 0.5})
        if distance > 20.0:
            layers.append({"name": "reverb", "volume": volume * 0.3})

        sound = {
            "event": event,
            "waveform": base["waveform"],
            "frequency": frequency,
            "duration": duration,
            "volume": volume,
            "distance": distance,
            "layers": layers,
        }
        self._history.append(("sound", event, distance))
        return sound

    # ---------------- 生命周期 ----------------

    def update(self, dt: float):
        transition = self._state.get("transition")
        if transition:
            transition["elapsed"] += dt
            if transition["elapsed"] >= transition["duration"]:
                self._state["emotion"] = dict(transition["to"])
                self._state["transition"] = None

    def reset(self):
        self._state = {}
        self._history = []
