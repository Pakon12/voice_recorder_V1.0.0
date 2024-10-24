# File: voice_recorder/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
import wave
from scipy import signal
from scipy.fft import fft, fftfreq
import pathlib

class AudioVisualizer:
    """
    A class for visualizing audio recordings with various analysis options.
    
    Features:
    - Waveform visualization
    - Spectrum analysis
    - Voice activity highlighting
    - Spectrogram visualization
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize AudioVisualizer.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 44100)
        """
        self.sample_rate = sample_rate
        # ใช้ style มาตรฐานที่สวยงาม
        plt.style.use('default')
        # ตั้งค่า style เพิ่มเติม
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _load_audio(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return samples and sample rate.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple containing audio samples and sample rate
        """
        try:
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(-1)
                samples = np.frombuffer(frames, dtype=np.int16)
                if wf.getnchannels() == 2:
                    samples = samples[::2]  # Convert stereo to mono
                samples = samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                return samples, wf.getframerate()
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            raise
    
    def _detect_voice_activity(self, samples: np.ndarray, 
                             frame_length: int = 2048,
                             hop_length: int = 512,
                             threshold: float = -40) -> np.ndarray:
        """
        Detect voice activity in audio samples.
        
        Args:
            samples: Audio samples
            frame_length: Length of each frame for analysis
            hop_length: Number of samples between frames
            threshold: Energy threshold in dB for voice detection
            
        Returns:
            Boolean array indicating voice activity
        """
        # Calculate energy in frames
        frames = []
        for i in range(0, len(samples) - frame_length, hop_length):
            frame = samples[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            frames.append(energy)
        
        # Convert to dB
        energy_db = 20 * np.log10(np.array(frames) + 1e-9)
        
        # Detect voice activity
        voice_activity = energy_db > threshold
        
        # Expand to sample length
        sample_activity = np.repeat(voice_activity, hop_length)
        if len(sample_activity) > len(samples):
            sample_activity = sample_activity[:len(samples)]
        elif len(sample_activity) < len(samples):
            sample_activity = np.pad(sample_activity, 
                                   (0, len(samples) - len(sample_activity)))
        
        return sample_activity
    
    def plot_waveform(self, audio_file: str, 
                     highlight_voice: bool = False,
                     threshold: float = -40,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None) -> None:
        """
        Plot audio waveform with optional voice activity highlighting.
        
        Args:
            audio_file: Path to audio file
            highlight_voice: Whether to highlight voice segments
            threshold: Energy threshold for voice detection
            title: Optional title for the plot
            save_path: Optional path to save the plot
        """
        # Load audio
        samples, sr = self._load_audio(audio_file)
        time = np.arange(len(samples)) / sr
        
        # Create figure
        plt.figure()
        
        # Plot waveform
        if highlight_voice:
            # Detect voice activity
            voice_activity = self._detect_voice_activity(samples, threshold=threshold)
            
            # Plot non-voice segments
            plt.plot(time[~voice_activity], samples[~voice_activity], 
                    color='gray', alpha=0.5, label='Non-voice')
            
            # Plot voice segments
            plt.plot(time[voice_activity], samples[voice_activity],
                    color='blue', alpha=0.8, label='Voice')
            plt.legend()
        else:
            plt.plot(time, samples, color='blue', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title or f'Waveform: {pathlib.Path(audio_file).name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_spectrum(self, audio_file: str,
                     window_size: int = 2048,
                     highlight_voice_freq: bool = True,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None) -> None:
        """
        Plot frequency spectrum of the audio.
        
        Args:
            audio_file: Path to audio file
            window_size: Size of the FFT window
            highlight_voice_freq: Whether to highlight human voice frequency range
            title: Optional title for the plot
            save_path: Optional path to save the plot
        """
        # Load audio
        samples, sr = self._load_audio(audio_file)
        
        # Calculate spectrum
        frequencies = fftfreq(window_size, 1/sr)
        spectrum = np.abs(fft(samples[:window_size]))
        
        # Plot only positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        spectrum = spectrum[positive_freq_mask]
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-9)
        
        # Create figure
        plt.figure()
        
        if highlight_voice_freq:
            # Plot full spectrum
            plt.plot(frequencies, spectrum_db, color='gray', alpha=0.5,
                    label='Full spectrum')
            
            # Highlight voice frequencies (roughly 85-255 Hz)
            voice_mask = (frequencies >= 85) & (frequencies <= 255)
            plt.plot(frequencies[voice_mask], spectrum_db[voice_mask],
                    color='blue', alpha=0.8, label='Voice range')
            plt.legend()
        else:
            plt.plot(frequencies, spectrum_db, color='blue', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title(title or f'Frequency Spectrum: {pathlib.Path(audio_file).name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def plot_spectrogram(self, audio_file: str,
                        window_size: int = 2048,
                        hop_length: int = 512,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Plot spectrogram of the audio.
        
        Args:
            audio_file: Path to audio file
            window_size: Size of the FFT window
            hop_length: Number of samples between frames
            title: Optional title for the plot
            save_path: Optional path to save the plot
        """
        # Load audio
        samples, sr = self._load_audio(audio_file)
        
        # Calculate spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            samples,
            sr,
            nperseg=window_size,
            noverlap=window_size-hop_length,
            scaling='spectrum'
        )
        
        # Convert to dB
        spectrogram_db = 10 * np.log10(spectrogram + 1e-9)
        
        # Create figure
        plt.figure()
        
        # Plot spectrogram
        plt.pcolormesh(times, frequencies, spectrogram_db,
                      shading='gouraud', cmap='viridis')
        
        # Customize plot
        plt.colorbar(label='Power (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (seconds)')
        plt.title(title or f'Spectrogram: {pathlib.Path(audio_file).name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def analyze_audio(self, audio_file: str,
                     output_dir: Optional[str] = None) -> None:
        """
        Perform comprehensive audio analysis and save all plots.
        
        Args:
            audio_file: Path to audio file
            output_dir: Directory to save plots (optional)
        """
        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create all visualizations
        print("Generating waveform plot...")
        self.plot_waveform(
            audio_file,
            highlight_voice=True,
            save_path=f"{output_dir}/waveform.png" if output_dir else None
        )
        
        print("Generating spectrum plot...")
        self.plot_spectrum(
            audio_file,
            highlight_voice_freq=True,
            save_path=f"{output_dir}/spectrum.png" if output_dir else None
        )
        
        print("Generating spectrogram...")
        self.plot_spectrogram(
            audio_file,
            save_path=f"{output_dir}/spectrogram.png" if output_dir else None
        )
        
        print("Analysis complete!")