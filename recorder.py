import pyaudio
import wave
import numpy as np
from typing import List, Optional, Tuple
import time
import keyboard
from scipy import signal

class AudioRecorder:
    def __init__(self,
                 chunk_size: int = 1024,
                 sample_format: int = pyaudio.paInt16,
                 channels: int = 1,
                 sample_rate: int = 44100,
                 voice_threshold: float = -25,
                 silence_duration: float = 1.0,
                 min_voice_duration: float = 0.1):
        self.chunk_size = chunk_size
        self.sample_format = sample_format
        self.channels = channels
        self.sample_rate = sample_rate
        self.voice_threshold = voice_threshold
        self.silence_duration = silence_duration
        self.min_voice_duration = min_voice_duration
        
        self._audio = None
        self._stream = None
        self._is_recording = False
        self._frames = []
        self._noise_floor = None

    def _calibrate_noise(self) -> None:
        """Calibrate noise floor by sampling ambient noise."""
        print("\n* Calibrating noise floor... Please stay quiet.")
        samples = []
        
        # Record ambient noise for 1 second
        for _ in range(int(self.sample_rate / self.chunk_size)):
            data = self._stream.read(self.chunk_size)
            samples.extend(np.frombuffer(data, dtype=np.int16))
            
        # Calculate noise floor
        self._noise_floor = float(np.mean(np.abs(samples))) * 1.2
        print("* Noise floor calibrated")
    
    def _reduce_noise(self, samples: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio samples."""
        # Create a copy of the array to avoid read-only issues
        cleaned = samples.copy()
        # Apply noise gate
        if self._noise_floor is not None:
            mask = np.abs(cleaned) < self._noise_floor
            cleaned[mask] = 0
        return cleaned

    def _detect_voice(self, data: bytes) -> Tuple[bool, float]:
        """
        Detect voice activity in audio data.
        Returns:
            Tuple[bool, float]: (is_voice_detected, activity_level)
        """
        # Convert bytes to numpy array
        samples = np.frombuffer(data, dtype=np.int16)
        
        # Apply noise reduction
        cleaned_samples = self._reduce_noise(samples)
        
        # Calculate RMS and convert to dB
        rms = np.sqrt(np.mean(cleaned_samples.astype(np.float64) ** 2))
        if rms == 0:
            return False, 0.0
            
        db = 20 * np.log10(max(rms, 1e-9))
        
        # Normalize activity level
        activity_level = (db - self.voice_threshold) / 40.0
        activity_level = max(0.0, min(1.0, activity_level))
        
        # Voice detection
        is_voice = db >= self.voice_threshold
        
        return is_voice, activity_level

    def _setup_audio(self) -> None:
        """Initialize audio interface."""
        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error setting up audio: {str(e)}")
            self._cleanup()
            raise

    def _save_recording(self, output_path: str) -> None:
        """Save recorded audio to file."""
        print("\n\n* Saving recording...")
        try:
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self._audio.get_sample_size(self.sample_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self._frames))
            print(f"* Saved to {output_path}")
        except Exception as e:
            print(f"Error saving recording: {str(e)}")
            raise

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        if hasattr(self, '_stream') and self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if hasattr(self, '_audio') and self._audio:
            self._audio.terminate()
        self._is_recording = False
        self._frames = []

    def start(self, output_path: str = None, monitor: bool = True) -> None:
        """Start recording with optional monitoring."""
        self._setup_audio()
        self._calibrate_noise()
        self._is_recording = True
        
        try:
            start_time = time.time()
            last_voice_time = time.time()
            consecutive_voice_chunks = 0
            
            # Activity visualization
            activity_chars = "▁▂▃▄▅▆▇█"
            
            while self._is_recording:
                if keyboard.is_pressed('q'):
                    print("\n\n* Recording stopped by user")
                    break
                
                # Read audio data
                data = self._stream.read(self.chunk_size)
                self._frames.append(data)
                
                # Detect voice
                is_voice, activity_level = self._detect_voice(data)
                
                if is_voice:
                    consecutive_voice_chunks += 1
                    if (consecutive_voice_chunks * self.chunk_size / self.sample_rate) >= self.min_voice_duration:
                        last_voice_time = time.time()
                else:
                    consecutive_voice_chunks = 0
                    if time.time() - last_voice_time > self.silence_duration:
                        print("\n\n* Recording stopped due to silence")
                        break
                
                # Update progress display
                if monitor:
                    current_time = time.time() - start_time
                    if is_voice:
                        idx = min(int(activity_level * len(activity_chars)), len(activity_chars) - 1)
                        activity_indicator = activity_chars[idx] * 3
                    else:
                        activity_indicator = activity_chars[0] * 3
                    print(f"\rRecording: {current_time:.1f}s | Activity: {activity_indicator}", end='')
            
            # Save recording if output path is provided
            if output_path:
                self._save_recording(output_path)
                
        except Exception as e:
            print(f"\nError during recording: {str(e)}")
            raise
        finally:
            self._cleanup()

    def record_until_silence(self, output_path: str, 
                           max_duration: float = None,
                           monitor: bool = True) -> str:
        """Record audio until silence is detected."""
        self.start(output_path, monitor)
        if max_duration:
            time.sleep(max_duration)
            self._is_recording = False
        return output_path

    def set_callback(self, event_type: str, callback) -> None:
        """
        Set callback function for specific events.
        
        Args:
            event_type: Type of event ('voice', 'silence', 'stop')
            callback: Function to call when event occurs
        """
        if event_type == 'voice':
            self.on_voice_detected = callback
        elif event_type == 'silence':
            self.on_silence_detected = callback
        elif event_type == 'stop':
            self.on_recording_stopped = callback