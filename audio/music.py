"""
游戏内程序化音乐生成与音频系统 - 核心模块

功能：
- 音乐理论：音阶、和弦、和弦进行、调式（大调/小调等）
- 旋律生成：基于音阶与节奏的程序化旋律
- 音频合成：振荡器（正弦/方波/锯齿波/三角波）+ ADSR 包络
- 自适应音乐：根据游戏场景切换情绪，并带平滑过渡
- 音效设计：分层音效、距离衰减、随机变化
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


# ---------------------------------------------------------------------------
# 音乐理论数据
# ---------------------------------------------------------------------------

# 调式（相对于根音的半音间隔）
MODES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}

# 和弦性质（相对于和弦根音的半音间隔）
CHORD_QUALITIES: Dict[str, List[int]] = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}

# 和弦进行（音阶级数 + 和弦性质）
CHORD_PROGRESSIONS: Dict[str, List[Tuple[int, str]]] = {
    "major": [(0, "major"), (5, "minor"), (3, "major"), (4, "major")],   # I-vi-IV-V
    "minor": [(0, "minor"), (5, "major"), (2, "major"), (4, "minor")],   # i-VI-III-v
    "dorian": [(0, "minor"), (3, "major"), (0, "minor"), (4, "minor")],
}

# 节奏池（以拍为单位的音符时长）
RHYTHM_POOL: List[float] = [0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0]

# 游戏场景 -> 情绪/音乐参数
SCENE_MOODS: Dict[str, Dict] = {
    "explore": {"emotion": "calm", "tempo": 80, "mode": "major",
                "waveform": "sine", "intensity": 0.3},
    "combat": {"emotion": "intense", "tempo": 150, "mode": "minor",
               "waveform": "sawtooth", "intensity": 0.9},
    "stealth": {"emotion": "tense", "tempo": 100, "mode": "dorian",
                "waveform": "triangle", "intensity": 0.5},
    "boss": {"emotion": "epic", "tempo": 170, "mode": "phrygian",
             "waveform": "square", "intensity": 1.0},
}

A4_MIDI = 69
A4_FREQ = 440.0


def midi_to_freq(midi: float) -> float:
    """MIDI 音高 -> 频率 (Hz)"""
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


@dataclass
class Config:
    sample_rate: int = 22050
    transition_time: float = 2.0   # 自适应音乐过渡时长（秒）
    base_key: int = 60             # 根音 MIDI（中央 C）


class ProceduralMusic:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.sample_rate = int(self.config.get("sample_rate", Config.sample_rate))
        self.transition_time = float(self.config.get("transition_time", Config.transition_time))
        self.base_key = int(self.config.get("base_key", Config.base_key))

        self._state = {}
        self._history = []

        # 自适应音乐状态
        self._scene = "explore"
        self._mood = dict(SCENE_MOODS["explore"])
        self._prev_mood: Optional[Dict] = None
        self._transition_progress = 1.0  # 1.0 表示过渡完成

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进自适应过渡等随时间变化的状态"""
        if self._transition_progress < 1.0:
            step = dt / max(self.transition_time, 1e-6)
            self._transition_progress = min(1.0, self._transition_progress + step)
            if self._transition_progress >= 1.0 and self._prev_mood is not None:
                self._prev_mood = None

    def reset(self):
        self._state = {}
        self._history = []
        self._scene = "explore"
        self._mood = dict(SCENE_MOODS["explore"])
        self._prev_mood = None
        self._transition_progress = 1.0

    # ------------------------------------------------------------------
    # 音乐理论
    # ------------------------------------------------------------------
    def get_scale(self, root: Optional[int] = None, mode: str = "major") -> List[int]:
        """返回指定调式的音阶（MIDI 音高列表）"""
        if mode not in MODES:
            raise ValueError(f"未知调式: {mode}")
        root = self.base_key if root is None else root
        return [root + interval for interval in MODES[mode]]

    def get_chord(self, root: int, quality: str = "major") -> List[int]:
        """返回和弦（MIDI 音高列表）"""
        if quality not in CHORD_QUALITIES:
            raise ValueError(f"未知和弦性质: {quality}")
        return [root + interval for interval in CHORD_QUALITIES[quality]]

    def chord_progression(self, root: Optional[int] = None,
                          mode: str = "major") -> List[List[int]]:
        """根据调式生成和弦进行，返回和弦列表"""
        root = self.base_key if root is None else root
        scale = self.get_scale(root, mode if mode in MODES else "major")
        degrees = CHORD_PROGRESSIONS.get(mode, CHORD_PROGRESSIONS["major"])
        progression = []
        for degree, quality in degrees:
            chord_root = scale[degree % len(scale)]
            progression.append(self.get_chord(chord_root, quality))
        return progression

    # ------------------------------------------------------------------
    # 旋律生成（音阶 + 节奏）
    # ------------------------------------------------------------------
    def generate_melody(self, mode: Optional[str] = None, length: int = 8,
                        seed: Optional[int] = None,
                        root: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        程序化生成旋律：音符取自调式音阶，时长取自节奏池。
        返回 [(midi_note, duration_beats), ...]
        """
        mode = mode or self._mood.get("mode", "major")
        scale = self.get_scale(root, mode)
        # 扩展到两个八度，旋律更有起伏
        scale = scale + [n + 12 for n in scale]
        rng = random.Random(seed)

        intensity = self._mood.get("intensity", 0.5)
        idx = rng.randrange(len(scale) // 2)
        melody = []
        for _ in range(length):
            # 随机游走：强度越高跳动越大
            max_step = 1 + int(round(intensity * 3))
            idx += rng.randint(-max_step, max_step)
            idx = max(0, min(len(scale) - 1, idx))
            note = scale[idx]
            duration = rng.choice(RHYTHM_POOL)
            melody.append((note, duration))
        self._history.append(("melody", mode, length))
        return melody

    # ------------------------------------------------------------------
    # 音频合成（振荡器 + 包络）
    # ------------------------------------------------------------------
    def oscillator(self, waveform: str, freq: float, t: float) -> float:
        """振荡器：生成单个采样点的波形值，范围 [-1, 1]"""
        phase = (freq * t) % 1.0
        if waveform == "sine":
            return math.sin(2.0 * math.pi * phase)
        if waveform == "square":
            return 1.0 if phase < 0.5 else -1.0
        if waveform == "sawtooth":
            return 2.0 * phase - 1.0
        if waveform == "triangle":
            return 4.0 * abs(phase - 0.5) - 1.0
        raise ValueError(f"未知波形: {waveform}")

    def envelope(self, t: float, duration: float, attack: float = 0.01,
                 decay: float = 0.1, sustain: float = 0.7,
                 release: float = 0.1) -> float:
        """ADSR 包络：起音-衰减-延音-释音"""
        if t < 0.0 or t > duration + release:
            return 0.0
        if t < attack:
            return t / max(attack, 1e-6)
        if t < attack + decay:
            frac = (t - attack) / max(decay, 1e-6)
            return 1.0 - frac * (1.0 - sustain)
        if t <= duration:
            return sustain
        # release
        frac = (t - duration) / max(release, 1e-6)
        return sustain * max(0.0, 1.0 - frac)

    def synthesize_note(self, freq: float, duration: float,
                        waveform: str = "sine",
                        sample_rate: Optional[int] = None) -> List[float]:
        """实时合成一个音符的采样（非预录音频）"""
        sr = sample_rate or self.sample_rate
        n = max(1, int(sr * duration))
        return [
            self.oscillator(waveform, freq, i / sr)
            * self.envelope(i / sr, duration)
            for i in range(n)
        ]

    def synthesize_melody(self, melody: List[Tuple[int, float]],
                          tempo: Optional[int] = None,
                          waveform: Optional[str] = None) -> List[float]:
        """把旋律合成为完整音频采样"""
        tempo = tempo or self._mood.get("tempo", 120)
        waveform = waveform or self._mood.get("waveform", "sine")
        beat_sec = 60.0 / tempo
        samples: List[float] = []
        for note, beats in melody:
            samples.extend(self.synthesize_note(
                midi_to_freq(note), beats * beat_sec, waveform))
        return samples

    # ------------------------------------------------------------------
    # 自适应音乐（情绪 + 过渡）
    # ------------------------------------------------------------------
    def set_scene(self, scene: str):
        """切换游戏场景，音乐情绪随之变化，并启动平滑过渡"""
        if scene not in SCENE_MOODS:
            raise ValueError(f"未知场景: {scene}")
        if scene == self._scene:
            return
        self._prev_mood = dict(self._mood)
        self._scene = scene
        self._mood = dict(SCENE_MOODS[scene])
        self._transition_progress = 0.0
        self._history.append(("scene", scene))

    def current_emotion(self) -> str:
        return self._mood["emotion"]

    def transition_value(self, key: str) -> float:
        """过渡中的数值参数（tempo/intensity 等），随 update 平滑插值"""
        target = float(self._mood[key])
        if self._prev_mood is None or self._transition_progress >= 1.0:
            return target
        start = float(self._prev_mood[key])
        # 平滑插值（smoothstep）
        p = self._transition_progress
        p = p * p * (3.0 - 2.0 * p)
        return start + (target - start) * p

    def is_transitioning(self) -> bool:
        return self._transition_progress < 1.0

    def generate_adaptive_music(self, length: int = 8,
                                seed: Optional[int] = None) -> Dict:
        """根据当前场景情绪生成音乐：旋律 + 和弦进行 + 合成音频"""
        mood = self._mood
        melody = self.generate_melody(mode=mood["mode"], length=length, seed=seed)
        chords = self.chord_progression(mode=mood["mode"])
        audio = self.synthesize_melody(melody)
        return {
            "scene": self._scene,
            "emotion": mood["emotion"],
            "tempo": mood["tempo"],
            "mode": mood["mode"],
            "melody": melody,
            "chords": chords,
            "audio": audio,
        }

    # ------------------------------------------------------------------
    # 音效设计（层次 + 变化）
    # ------------------------------------------------------------------
    def design_sound(self, event: str, distance: float = 0.0,
                     seed: Optional[int] = None) -> Dict:
        """
        设计一个音效：
        - 层次：基础层 + 若干细节层（数量随事件强度）
        - 距离衰减：越远音量越小、高频越少
        - 变化：同一事件每次生成音高/时长略有不同
        """
        rng = random.Random(seed if seed is not None else time.time_ns())
        event_profiles = {
            "footstep": {"layers": 2, "base_freq": 120.0, "duration": 0.15},
            "explosion": {"layers": 4, "base_freq": 60.0, "duration": 1.2},
            "ui_click": {"layers": 1, "base_freq": 800.0, "duration": 0.05},
            "sword_hit": {"layers": 3, "base_freq": 300.0, "duration": 0.4},
        }
        profile = event_profiles.get(event, {"layers": 2, "base_freq": 220.0,
                                             "duration": 0.3})

        # 距离衰减：音量随距离反比下降，低通截止频率随距离下降
        attenuation = 1.0 / (1.0 + max(distance, 0.0))
        lowpass = max(200.0, 8000.0 * attenuation)

        # 随机变化：音高 ±10%，时长 ±15%
        pitch_var = 1.0 + rng.uniform(-0.1, 0.1)
        dur_var = 1.0 + rng.uniform(-0.15, 0.15)

        waveforms = ["sine", "triangle", "square", "sawtooth"]
        layers = []
        for i in range(profile["layers"]):
            layers.append({
                "waveform": waveforms[i % len(waveforms)],
                "freq": profile["base_freq"] * pitch_var * (i + 1),
                "gain": attenuation / (i + 1),
                "duration": profile["duration"] * dur_var,
            })

        sound = {
            "event": event,
            "layers": layers,
            "layer_count": len(layers),
            "distance": distance,
            "attenuation": attenuation,
            "lowpass_cutoff": lowpass,
            "pitch_variation": pitch_var,
        }
        self._history.append(("sound", event))
        return sound
