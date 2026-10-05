"""
游戏内程序化音乐生成与音频系统 - 验收测试
运行方式：python -m pytest tests/test_procedural_music.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.music import ProceduralMusic, midi_to_freq, MODES


class TestProceduralMusic:
    def setup_method(self):
        self.system = ProceduralMusic()

    # ---- 旋律生成：音阶 ----
    def test_case_01(self):
        """音阶：大调音阶间隔正确（全全半全全全半）"""
        scale = self.system.get_scale(60, "major")
        assert scale == [60, 62, 64, 65, 67, 69, 71]

    def test_case_02(self):
        """旋律生成：所有音符都落在调式音阶内"""
        melody = self.system.generate_melody(mode="minor", length=32, seed=1)
        assert len(melody) == 32
        scale_pitches = {(60 + i) % 12 for i in MODES["minor"]}
        for note, _ in melody:
            assert note % 12 in scale_pitches

    # ---- 旋律生成：节奏 ----
    def test_case_03(self):
        """旋律生成：音符时长不全都相同（有节奏变化）"""
        melody = self.system.generate_melody(mode="major", length=32, seed=7)
        durations = {d for _, d in melody}
        assert len(durations) > 1
        assert all(d > 0 for _, d in melody)

    def test_case_04(self):
        """旋律生成：相同种子可复现，不同种子产生不同旋律"""
        m1 = self.system.generate_melody(length=16, seed=42)
        m2 = self.system.generate_melody(length=16, seed=42)
        m3 = self.system.generate_melody(length=16, seed=43)
        assert m1 == m2
        assert m1 != m3

    # ---- 音频合成：振荡器 ----
    def test_case_05(self):
        """振荡器：正弦/方波/锯齿波波形各不相同且取值正确"""
        t = 0.0
        assert abs(self.system.oscillator("sine", 440, t)) < 1e-9
        assert self.system.oscillator("square", 440, t + 0.25 / 440) == 1.0
        assert self.system.oscillator("square", 440, t + 0.75 / 440) == -1.0
        assert abs(self.system.oscillator("sawtooth", 440, t + 0.5 / 440)) < 1e-9
        s = self.system.oscillator("sine", 440, 0.1)
        q = self.system.oscillator("square", 440, 0.1)
        w = self.system.oscillator("sawtooth", 440, 0.1)
        assert len({round(s, 6), round(q, 6), round(w, 6)}) == 3

    # ---- 音频合成：包络 ----
    def test_case_06(self):
        """包络：有起音（从0上升）和衰减/释音（结尾回到0）"""
        dur = 1.0
        assert self.system.envelope(0.0, dur) == 0.0
        mid_attack = self.system.envelope(0.005, dur, attack=0.01)
        assert 0.0 < mid_attack < 1.0
        assert self.system.envelope(0.5, dur) == pytest.approx(0.7)  # sustain
        assert self.system.envelope(dur + 0.2, dur, release=0.1) == 0.0

    def test_case_07(self):
        """合成：synthesize_note 实时生成采样，长度正确且非静音"""
        samples = self.system.synthesize_note(440.0, 0.1, "sine", sample_rate=8000)
        assert len(samples) == 800
        assert max(abs(s) for s in samples) > 0.5
        # 不同频率合成结果不同（不是预录固定音频）
        other = self.system.synthesize_note(220.0, 0.1, "sine", sample_rate=8000)
        assert samples != other

    # ---- 音乐理论：和弦与调式 ----
    def test_case_08(self):
        """和弦：大三和弦与小三和弦结构正确"""
        assert self.system.get_chord(60, "major") == [60, 64, 67]
        assert self.system.get_chord(60, "minor") == [60, 63, 67]

    def test_case_09(self):
        """和弦进行：生成多个和弦且都取自调内音"""
        prog = self.system.chord_progression(60, "major")
        assert len(prog) >= 3
        scale_pitches = {n % 12 for n in self.system.get_scale(60, "major")}
        for chord in prog:
            assert len(chord) >= 3
            for note in chord:
                assert note % 12 in scale_pitches

    def test_case_10(self):
        """调式：大调与小调音阶不同（第三级音不同）"""
        major = self.system.get_scale(60, "major")
        minor = self.system.get_scale(60, "minor")
        assert major != minor
        assert major[2] - 60 == 4   # 大三度
        assert minor[2] - 60 == 3   # 小三度

    # ---- 自适应音乐：情绪 ----
    def test_case_11(self):
        """自适应：战斗与探索场景情绪/速度/调式不同"""
        explore = self.system.generate_adaptive_music(seed=1)
        assert explore["emotion"] == "calm"
        self.system.set_scene("combat")
        combat = self.system.generate_adaptive_music(seed=1)
        assert combat["emotion"] == "intense"
        assert combat["tempo"] != explore["tempo"]
        assert combat["mode"] != explore["mode"]

    # ---- 自适应音乐：过渡 ----
    def test_case_12(self):
        """过渡：场景切换是渐变的，不是瞬间完成"""
        self.system.set_scene("combat")
        assert self.system.is_transitioning()
        t0 = self.system.transition_value("tempo")
        assert t0 == pytest.approx(80.0)  # 起点仍是探索的速度
        self.system.update(1.0)           # 过渡一半
        t_mid = self.system.transition_value("tempo")
        assert 80.0 < t_mid < 150.0
        self.system.update(10.0)          # 过渡完成
        assert not self.system.is_transitioning()
        assert self.system.transition_value("tempo") == pytest.approx(150.0)

    # ---- 音效设计：层次 ----
    def test_case_13(self):
        """音效层次：多层音效，且距离越远衰减越大"""
        near = self.system.design_sound("explosion", distance=0.0, seed=5)
        far = self.system.design_sound("explosion", distance=10.0, seed=5)
        assert near["layer_count"] >= 2
        assert len(near["layers"]) == near["layer_count"]
        assert far["attenuation"] < near["attenuation"]
        assert far["lowpass_cutoff"] < near["lowpass_cutoff"]
        for layer in far["layers"]:
            assert layer["gain"] <= far["attenuation"]

    # ---- 音效设计：变化 ----
    def test_case_14(self):
        """音效变化：同一动作多次生成的音效不完全相同"""
        s1 = self.system.design_sound("sword_hit", seed=11)
        s2 = self.system.design_sound("sword_hit", seed=22)
        assert s1["pitch_variation"] != s2["pitch_variation"]
        f1 = [l["freq"] for l in s1["layers"]]
        f2 = [l["freq"] for l in s2["layers"]]
        assert f1 != f2

    # ---- 综合 ----
    def test_case_15(self):
        """综合：自适应音乐产出旋律+和弦进行+合成音频"""
        result = self.system.generate_adaptive_music(length=8, seed=3)
        assert len(result["melody"]) == 8
        assert len(result["chords"]) >= 3
        assert len(result["audio"]) > 0
        assert all(isinstance(s, float) for s in result["audio"])
        # 音频采样来自合成，振幅在合理范围内
        assert max(abs(s) for s in result["audio"]) <= 1.0
        # midi_to_freq 换算正确
        assert midi_to_freq(69) == pytest.approx(440.0)
        assert midi_to_freq(60) == pytest.approx(261.63, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
