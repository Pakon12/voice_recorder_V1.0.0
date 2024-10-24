# File: voice_recorder/examples/__init__.py
"""
Voice Recorder Examples Package

This package contains example scripts demonstrating various ways to use the voice_recorder library.

Available examples:
1. Basic Recording - Simple audio recording with voice detection
2. Advanced Recording - Recording with custom parameters and callbacks
3. Audio Analysis - Various ways to analyze and visualize audio
4. Real-time Monitoring - Recording with real-time voice activity display
5. Batch Processing - Processing multiple audio files
"""

from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from voice_recorder.recorder import AudioRecorder
from voice_recorder.visualizer import AudioVisualizer

__all__ = ['basic_recording', 'advanced_recording', 'audio_analysis', 
           'realtime_monitor', 'batch_processor']

# Example 1: Basic Recording
def basic_recording():
    """Basic example of recording audio with voice detection."""
    recorder = AudioRecorder()
    output_file = "basic_recording.wav"
    
    print("Starting basic recording...")
    print("Speak into the microphone (will stop after 1 second of silence)")
    print("Press 'q' to stop manually")
    
    recorder.record_until_silence(output_file)
    print(f"Recording saved to {output_file}")

# Example 2: Advanced Recording
def advanced_recording():
    """Advanced recording with custom settings and callbacks."""
    # Custom settings
    recorder = AudioRecorder(
        voice_threshold=-30,      # More sensitive voice detection
        silence_duration=2.0,     # Longer silence before stopping
        min_voice_duration=0.3    # Longer minimum voice duration
    )
    
    # Callback functions
    def on_voice():
        print("\nVoice detected!", end='', flush=True)
    
    def on_silence():
        print("\nSilence detected!", end='', flush=True)
    
    def on_stop():
        print("\nRecording stopped")
    
    # Set callbacks
    recorder.set_callback('voice', on_voice)
    recorder.set_callback('silence', on_silence)
    recorder.set_callback('stop', on_stop)
    
    print("Starting advanced recording...")
    recorder.record_until_silence("advanced_recording.wav")

# Example 3: Audio Analysis
def audio_analysis(audio_file: str = "recording.wav"):
    """Demonstrate various audio analysis features."""
    visualizer = AudioVisualizer()
    
    # 1. Basic waveform
    print("\nPlotting basic waveform...")
    visualizer.plot_waveform(audio_file, highlight_voice=False)
    
    # 2. Waveform with voice detection
    print("\nPlotting waveform with voice detection...")
    visualizer.plot_waveform(audio_file, highlight_voice=True)
    
    # 3. Frequency spectrum
    print("\nPlotting frequency spectrum...")
    visualizer.plot_spectrum(audio_file, highlight_voice_freq=True)
    
    # 4. Spectrogram
    print("\nPlotting spectrogram...")
    visualizer.plot_spectrogram(audio_file)
    
    # 5. Complete analysis
    print("\nPerforming complete analysis...")
    visualizer.analyze_audio(audio_file, output_dir="analysis_results")

# Example 4: Real-time Monitoring
def realtime_monitor():
    """Record audio with real-time monitoring."""
    recorder = AudioRecorder(voice_threshold=-35)
    
    def on_voice():
        print("🎤", end='', flush=True)  # Microphone emoji when voice detected
    
    def on_silence():
        print("⚪", end='', flush=True)   # Circle emoji when silence detected
    
    recorder.set_callback('voice', on_voice)
    recorder.set_callback('silence', on_silence)
    
    print("Starting real-time monitoring...")
    print("Voice indicators: 🎤 = voice, ⚪ = silence")
    recorder.record_until_silence("monitored_recording.wav")

# Example 5: Batch Processing
def batch_processor(input_dir: str = "recordings", 
                   output_dir: str = "analyzed_recordings"):
    """Process multiple audio files in a directory."""
    from pathlib import Path
    
    # Create visualizer
    visualizer = AudioVisualizer()
    
    # Process each WAV file in the input directory
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for audio_file in input_path.glob("*.wav"):
        print(f"\nProcessing {audio_file.name}...")
        
        # Create output directory for this file
        file_output_dir = output_path / audio_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        # Analyze audio
        visualizer.analyze_audio(
            str(audio_file),
            output_dir=str(file_output_dir)
        )

def main():
    """Run all examples."""
    print("Voice Recorder Examples")
    print("======================")
    
    while True:
        print("\nAvailable examples:")
        print("1. Basic Recording")
        print("2. Advanced Recording")
        print("3. Audio Analysis")
        print("4. Real-time Monitoring")
        print("5. Batch Processing")
        print("0. Exit")
        
        choice = input("\nSelect an example to run (0-5): ")
        
        if choice == "1":
            basic_recording()
        elif choice == "2":
            advanced_recording()
        elif choice == "3":
            audio_file = input("Enter audio file path (or press Enter for default): ")
            if audio_file.strip():
                audio_analysis(audio_file)
            else:
                audio_analysis()
        elif choice == "4":
            realtime_monitor()
        elif choice == "5":
            input_dir = input("Enter input directory (or press Enter for default): ")
            output_dir = input("Enter output directory (or press Enter for default): ")
            if input_dir.strip() and output_dir.strip():
                batch_processor(input_dir, output_dir)
            else:
                batch_processor()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()