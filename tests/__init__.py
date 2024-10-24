# File: voice_recorder/tests/__init__.py
import unittest
import os
import numpy as np
import tempfile
from pathlib import Path
import wave
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from voice_recorder.recorder import AudioRecorder
from voice_recorder.visualizer import AudioVisualizer

class TestAudioRecorder(unittest.TestCase):
    """Test cases for AudioRecorder class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.recorder = AudioRecorder(
            voice_threshold=-30,
            silence_duration=1.0,
            min_voice_duration=0.1
        )
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_recording.wav")
        
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test AudioRecorder initialization."""
        self.assertEqual(self.recorder.voice_threshold, -30)
        self.assertEqual(self.recorder.silence_duration, 1.0)
        self.assertEqual(self.recorder.min_voice_duration, 0.1)
        self.assertFalse(self.recorder._is_recording)
    
    def test_callback_registration(self):
        """Test callback registration."""
        def dummy_callback():
            pass
        
        self.recorder.set_callback('voice', dummy_callback)
        self.recorder.set_callback('silence', dummy_callback)
        self.recorder.set_callback('stop', dummy_callback)
        
        self.assertEqual(self.recorder.on_voice_detected, dummy_callback)
        self.assertEqual(self.recorder.on_silence_detected, dummy_callback)
        self.assertEqual(self.recorder.on_recording_stopped, dummy_callback)
    
    def test_create_wav_file(self):
        """Test creation of WAV file."""
        # Create a short test recording
        duration = 0.1  # seconds
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        with wave.open(self.test_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())
        
        # Verify file exists and has correct properties
        self.assertTrue(os.path.exists(self.test_file))
        with wave.open(self.test_file, 'rb') as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), sample_rate)
    
    def test_voice_detection(self):
        """Test voice activity detection."""
        # Create test signal with known voice activity
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with alternating voice and silence
        voice = np.sin(2 * np.pi * 440 * t) * 0.5  # Voice
        silence = np.zeros_like(t) * 0.01  # Silence
        
        signal = np.concatenate([voice[:len(voice)//2], silence[len(silence)//2:]])
        signal_bytes = (signal * 32767).astype(np.int16).tobytes()
        
        # Test voice detection
        is_voice, level = self.recorder._detect_voice(signal_bytes)
        self.assertIsInstance(is_voice, bool)
        self.assertIsInstance(level, float)
        self.assertTrue(0 <= level <= 1)

class TestAudioVisualizer(unittest.TestCase):
    """Test cases for AudioVisualizer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.visualizer = AudioVisualizer()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_audio.wav")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create test audio file
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        with wave.open(self.test_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)
        os.rmdir(self.temp_dir)
    
    def test_load_audio(self):
        """Test audio file loading."""
        samples, sr = self.visualizer._load_audio(self.test_file)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(sr, 44100)
        self.assertTrue(len(samples) > 0)
    
    def test_plot_generation(self):
        """Test generation of all plot types."""
        # Test waveform plot
        self.visualizer.plot_waveform(self.test_file, save_path=None)
        
        # Test spectrum plot
        self.visualizer.plot_spectrum(self.test_file, save_path=None)
        
        # Test spectrogram plot
        self.visualizer.plot_spectrogram(self.test_file, save_path=None)
    
    def test_complete_analysis(self):
        """Test complete audio analysis with file saving."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Run complete analysis
        self.visualizer.analyze_audio(self.test_file, self.output_dir)
        
        # Check if output files were created
        expected_files = ['waveform.png', 'spectrum.png', 'spectrogram.png']
        for file in expected_files:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, file)))
    
    def test_error_handling(self):
        """Test error handling for invalid files."""
        # Test with non-existent file
        with self.assertRaises(Exception):
            self.visualizer.plot_waveform("nonexistent.wav")
        
        # Test with invalid file
        invalid_file = os.path.join(self.temp_dir, "invalid.wav")
        with open(invalid_file, 'w') as f:
            f.write("Not a WAV file")
        
        with self.assertRaises(Exception):
            self.visualizer.plot_waveform(invalid_file)
        
        os.remove(invalid_file)

def run_tests():
    """Run all tests."""
    print("Running Voice Recorder Tests")
    print("===========================")
    unittest.main(argv=[''], verbosity=2)

if __name__ == "__main__":
    run_tests()