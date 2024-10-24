# Voice Recorder Library

ไลบรารีสำหรับบันทึกเสียงและวิเคราะห์เสียงอัตโนมัติ พร้อมระบบตรวจจับเสียงพูดและการแสดงผลที่หลากหลาย

## ✨ คุณสมบัติ

- 🎤 บันทึกเสียงอัตโนมัติ
- 🔍 ตรวจจับเสียงพูดอัตโนมัติ
- 📊 วิเคราะห์และแสดงผลเสียงหลากหลายรูปแบบ
- 📈 สร้างกราฟและแผนภาพแสดงผลการวิเคราะห์
- ⚡ ทำงานแบบ Real-time
- 🔧 ปรับแต่งค่าต่างๆ ได้ง่าย

## 📦 การติดตั้ง

1. ติดตั้งผ่าน pip:
```bash
pip install voice-recorder
```

2. ติดตั้ง dependencies ที่จำเป็น:
```bash
pip install numpy matplotlib scipy pyaudio keyboard
```

## 🚀 การใช้งานเบื้องต้น

### การบันทึกเสียงพื้นฐาน

```python
from voice_recorder import AudioRecorder

# สร้าง recorder
recorder = AudioRecorder()

# บันทึกเสียงจนกว่าจะตรวจพบความเงียบ
recorder.record_until_silence("recording.wav")
```

### การบันทึกเสียงขั้นสูง

```python
# ปรับแต่งค่าต่างๆ
recorder = AudioRecorder(
    voice_threshold=-30,      # ความไวในการตรวจจับเสียง
    silence_duration=1.5,     # เวลารอก่อนหยุดเมื่อไม่มีเสียง
    min_voice_duration=0.2    # ระยะเวลาขั้นต่ำของเสียง
)

# กำหนด callbacks
def on_voice():
    print("ตรวจพบเสียง!")

def on_silence():
    print("ตรวจพบความเงียบ!")

recorder.set_callback('voice', on_voice)
recorder.set_callback('silence', on_silence)

# เริ่มบันทึก
recorder.start("output.wav")
```

### การวิเคราะห์เสียง

```python
from voice_recorder import AudioVisualizer

# สร้าง visualizer
visualizer = AudioVisualizer()

# แสดงรูปคลื่นเสียง
visualizer.plot_waveform("recording.wav", highlight_voice=True)

# แสดงสเปกตรัมความถี่
visualizer.plot_spectrum("recording.wav")

# แสดง spectrogram
visualizer.plot_spectrogram("recording.wav")

# วิเคราะห์แบบครบถ้วน
visualizer.analyze_audio("recording.wav", output_dir="analysis_results")
```

## 📝 คำอธิบายคลาสหลัก

### AudioRecorder

คลาสสำหรับบันทึกเสียงพร้อมระบบตรวจจับเสียงอัตโนมัติ

#### พารามิเตอร์สำคัญ:
- `voice_threshold` (-40 ถึง 0): ความไวในการตรวจจับเสียง
- `silence_duration`: เวลารอก่อนหยุดเมื่อไม่มีเสียง (วินาที)
- `min_voice_duration`: ระยะเวลาขั้นต่ำของเสียง (วินาที)

#### เมธอดสำคัญ:
- `start()`: เริ่มบันทึกเสียง
- `stop()`: หยุดบันทึก
- `record_until_silence()`: บันทึกจนกว่าจะตรวจพบความเงียบ
- `set_callback()`: กำหนดฟังก์ชันเมื่อเกิดเหตุการณ์

### AudioVisualizer

คลาสสำหรับวิเคราะห์และแสดงผลเสียงในรูปแบบต่างๆ

#### เมธอดสำคัญ:
- `plot_waveform()`: แสดงรูปคลื่นเสียง
- `plot_spectrum()`: แสดงสเปกตรัมความถี่
- `plot_spectrogram()`: แสดง spectrogram
- `analyze_audio()`: วิเคราะห์เสียงแบบครบถ้วน

## 📚 ตัวอย่างการใช้งาน

ดูตัวอย่างเพิ่มเติมได้ในโฟลเดอร์ `examples/`:

1. `basic_recording.py`: การบันทึกเสียงพื้นฐาน
2. `advanced_recording.py`: การบันทึกเสียงขั้นสูง
3. `audio_analysis.py`: การวิเคราะห์เสียง
4. `realtime_monitor.py`: การตรวจจับเสียงแบบ real-time
5. `batch_processor.py`: การประมวลผลไฟล์เสียงจำนวนมาก

## ⚙️ การทดสอบ

รันการทดสอบทั้งหมด:
```bash
python -m voice_recorder.tests
```

## 🤝 การมีส่วนร่วม

1. Fork โปรเจค
2. สร้าง branch ใหม่: `git checkout -b feature/amazing-feature`
3. Commit การเปลี่ยนแปลง: `git commit -m 'Add amazing feature'`
4. Push ไปยัง branch: `git push origin feature/amazing-feature`
5. เปิด Pull Request

## 📄 ลิขสิทธิ์

โปรเจคนี้ใช้ลิขสิทธิ์ MIT License - ดูรายละเอียดในไฟล์ [LICENSE](LICENSE)

## 📧 ติดต่อ

- อีเมล: pkorn8394@gmail.com
- เว็บไซต์: https://https://portfolio-pakorn.vercel.app/
- GitHub: https://github.com/Pakon12
## 🙏 ขอขอบคุณ

- PyAudio สำหรับการบันทึกเสียง
- Matplotlib สำหรับการแสดงผลกราฟ
- SciPy สำหรับการประมวลผลสัญญาณ
- NumPy สำหรับการคำนวณ

## 📝 หมายเหตุ

- ต้องติดตั้ง PyAudio ก่อนใช้งาน
- ทดสอบบน Python 3.7 ขึ้นไป
- ต้องการไมโครโฟนสำหรับการบันทึกเสียง