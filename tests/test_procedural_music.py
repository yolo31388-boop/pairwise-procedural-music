"""
游戏内程序化音乐生成与音频系统 - 验收测试
运行方式：python -m pytest tests/test_procedural_music.py -q
共 15 个测试用例
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.music import ProceduralMusic, MODES, NOTE_OFFSETS, NOTE_NAMES


class TestProceduralMusic:
    def setup_method(self):
        self.system = ProceduralMusic()

    # ---------- 旋律生成：音阶、节奏、可复现 ----------

    def test_case_01(self):
        melody = self.system.generate_melody(length=16, key="C", mode="major", seed=1)
        scale = {NOTE_NAMES[(NOTE_OFFSETS["C"] + iv) % 12] for iv in MODES["major"]}
        assert len(melody) == 16
        assert all(note["note"] in scale for note in melody)

    def test_case_02(self):
        melody = self.system.generate_melody(length=16, seed=2)
        durations = {note["duration"] for note in melody}
        assert len(durations) > 1
        assert all(note["duration"] > 0 for note in melody)

    def test_case_03(self):
        first = self.system.generate_melody(length=8, key="A", mode="minor", seed=7)
        second = self.system.generate_melody(length=8, key="A", mode="minor", seed=7)
        assert first == second
        assert all(note["frequency"] > 0 for note in first)

    # ---------- 音频合成：振荡器、波形、包络 ----------

    def test_case_04(self):
        samples = self.system.audio_synthesis(
            "sine", frequency=1.0, duration=1.0, sample_rate=8,
            envelope=(0.0, 0.0, 1.0, 0.0),
        )
        assert len(samples) == 8
        for i, value in enumerate(samples):
            assert value == pytest.approx(math.sin(2.0 * math.pi * i / 8))

    def test_case_05(self):
        square = self.system.audio_synthesis(
            "square", frequency=1.0, duration=1.0, sample_rate=8,
            envelope=(0.0, 0.0, 1.0, 0.0),
        )
        saw = self.system.audio_synthesis(
            "sawtooth", frequency=1.0, duration=1.0, sample_rate=8,
            envelope=(0.0, 0.0, 1.0, 0.0),
        )
        assert set(round(v) for v in square if v != 0) == {-1, 1}
        assert max(saw) == pytest.approx(1.0, abs=0.3)
        assert min(saw) == pytest.approx(-1.0, abs=0.2)
        assert any(abs(a - b) > 0.1 for a, b in zip(square, saw))

    def test_case_06(self):
        samples = self.system.audio_synthesis(
            "sine", frequency=440.0, duration=1.0, sample_rate=2000,
            envelope=(0.1, 0.1, 0.6, 0.2),
        )
        assert abs(samples[0]) < 1e-6
        assert abs(samples[-1]) < 0.05
        assert max(samples) > 0.9

    # ---------- 音乐理论：调式、和弦、和弦进行 ----------

    def test_case_07(self):
        theory = self.system.music_theory(key="C", mode="major")
        assert theory["chord_progression"][0] == ["C", "E", "G"]
        assert theory["chord_progression"][1] == ["G", "B", "D"]
        assert theory["chord_progression"][2] == ["A", "C", "E"]
        assert theory["chord_progression"][3] == ["F", "A", "C"]

    def test_case_08(self):
        major = self.system.music_theory(key="C", mode="major")
        minor = self.system.music_theory(key="C", mode="minor")
        assert major["intervals"] != minor["intervals"]
        assert major["scale"][2] == "E"
        assert minor["scale"][2] == "D#"
        assert major["is_major"] is True
        assert minor["is_major"] is False

    def test_case_09(self):
        theory = self.system.music_theory(key="G", mode="major")
        for numeral, chord in theory["chords"].items():
            assert len(chord) == 3
            assert all(note in theory["scale"] for note in chord)
        assert theory["chords"]["I"] == ["G", "B", "D"]
        assert theory["chords"]["vii"] == ["F#", "A", "C"]

    # ---------- 自适应音乐：情绪、过渡、过渡完成 ----------

    def test_case_10(self):
        exploration = self.system.adaptive_music("exploration")
        self.system.adaptive_music("combat")
        self.system.update(2.0)
        combat = self.system.adaptive_music("combat")
        assert exploration["emotion"]["mood"] != combat["emotion"]["mood"]
        assert combat["emotion"]["tempo"] > exploration["emotion"]["tempo"]
        assert combat["emotion"]["intensity"] > exploration["emotion"]["intensity"]

    def test_case_11(self):
        self.system.adaptive_music("exploration", transition_time=2.0)
        state = self.system.adaptive_music("combat", transition_time=2.0)
        assert state["transitioning"] is True
        self.system.update(1.0)
        mid = self.system.adaptive_music("combat")
        assert mid["transitioning"] is True
        assert 0.0 < mid["progress"] < 1.0
        exploration_intensity = 0.4
        combat_intensity = 0.9
        assert exploration_intensity < mid["emotion"]["intensity"] < combat_intensity

    def test_case_12(self):
        self.system.adaptive_music("exploration", transition_time=1.0)
        self.system.adaptive_music("boss", transition_time=1.0)
        self.system.update(1.0)
        state = self.system.adaptive_music("boss")
        assert state["transitioning"] is False
        assert state["progress"] == 1.0
        assert state["emotion"]["intensity"] == 1.0
        assert state["emotion"]["tempo"] == 160.0

    # ---------- 音效设计：层次、变化、可复现/重置 ----------

    def test_case_13(self):
        near = self.system.sound_design("explosion", distance=0.0, seed=1)
        far = self.system.sound_design("explosion", distance=30.0, seed=1)
        assert len(near["layers"]) == 1
        assert len(far["layers"]) == 3
        assert far["volume"] < near["volume"]
        assert {layer["name"] for layer in far["layers"]} == {"dry", "reflection", "reverb"}

    def test_case_14(self):
        first = self.system.sound_design("sword_hit")
        second = self.system.sound_design("sword_hit")
        assert first["frequency"] != second["frequency"] or first["duration"] != second["duration"]

    def test_case_15(self):
        first = self.system.sound_design("footstep", distance=3.0, seed=42)
        second = self.system.sound_design("footstep", distance=3.0, seed=42)
        assert first == second
        self.system.adaptive_music("combat")
        self.system.reset()
        state = self.system.adaptive_music("combat")
        assert state["transitioning"] is False
        assert state["progress"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
