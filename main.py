from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import subprocess
from pathlib import Path
from fastapi.responses import FileResponse
from pydub import AudioSegment
import datetime
import librosa
import soundfile as sf
import numpy as np
# import scipy.io.wavfile as wav
# import scipy.fftpack as fft
app = FastAPI()

# CLI Binaries Paths
EMBEDDER_CLI = "binaries/SitMarkAudio2MEmbedderCLI"
DETECTOR_CLI = "binaries/SitMarkAudio2MDetectorCLI"

# Ensure output directory exists
OUTPUT_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "detection_summary.txt")

# if not os.path.exists(SUMMARY_FILE):
#     with open(SUMMARY_FILE, "w") as summary:
#         summary.write("CRC ok   | CRC failed | Filename\n")
#         summary.write("-" * 50 + "\n")



app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

@app.get("/")
async def home(request: Request):
    files = os.listdir(OUTPUT_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "files": files})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    raise HTTPException(status_code=404, detail="File not found")
@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "File deleted successfully"}
    raise HTTPException(status_code=404, detail="File not found")

def binary_to_text(binary: str) -> str:
    print('--------------------1-------------')
    """Convert binary string to text."""
    try:
        return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))
    except ValueError:
        return "Invalid binary format"

@app.post("/convert-binary/")
async def convert_binary_to_text(request: Request, binary: str = Form(...)):
    print('--------------------2-------------')

    converted_text = binary_to_text(binary)
    files = os.listdir(OUTPUT_DIR)  # Fetch uploaded files
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "converted_text": converted_text, 
        "files": files
    })


def text_to_binary(text: str, max_length: int = 64) -> str:
    print('--------------------4-------------')

    """Convert text to binary and ensure it fits max_length."""
    binary_str = ''.join(format(ord(char), '08b') for char in text)
    return binary_str[:max_length]  # Truncate if longer
def get_audio_loudness(audio_path):
    print('--------------------5-------------')
    
    """Analyze audio loudness using SoX."""
    result = subprocess.run(["sox", audio_path, "-n", "stat"], stderr=subprocess.PIPE, text=True)
    for line in result.stderr.split("\n"):
        if "RMS lev dB" in line:
            return float(line.split()[-1])
    return -40  # Default to very quiet if not detected

def get_dominant_frequency(audio_path):
    print('--------------------6-------------')

    """Estimate dominant frequency using SoX."""
    result = subprocess.run(["sox", audio_path, "-n", "stat"], stderr=subprocess.PIPE, text=True)
    for line in result.stderr.split("\n"):
        if "Frequency" in line:
            return int(line.split()[-1])
    return 3000  # Default to mid-range frequency


@app.post("/embed/")
async def embed_watermark(audio: UploadFile = File(...), watermark_message: str = Form(...)):
    try:
        binary_watermark = text_to_binary(watermark_message, max_length=64)
        input_audio_path = os.path.join(OUTPUT_DIR, audio.filename)
        output_audio = os.path.join(OUTPUT_DIR, "marked_" + audio.filename)

        with open(input_audio_path, "wb") as f:
            f.write(audio.file.read())

        if not input_audio_path.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV format supported")

        y, sr = librosa.load(input_audio_path, sr=None)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        loud_regions = np.where(rms > np.percentile(rms, 75))[0]

        if len(loud_regions) == 0:
            raise HTTPException(status_code=400, detail="No loud region found in audio.")

        embed_command = [
            EMBEDDER_CLI, "-i", input_audio_path, "-o", output_audio,
            "--wm-message", binary_watermark, "--wm-length", "64",
            "--ecc-mode", "1", "--min-freq", "2000", "--max-freq", "8000",
            "--wm-strength-target", "20", "--wm-strength-min", "10",
            "--redundancy", "2", "--filter-normalize"
        ]
        subprocess.run(embed_command, check=True)

        return {"message": "Watermark embedded successfully", "output_file": output_audio}
    
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to embed watermark")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def update_detection_summary(filename: str, crc_ok: int, crc_failed: int, summary_file: str):
    line = f"{crc_ok:<10}{crc_failed:<12}{filename}\n"
    header = f"{'CRC ok':<10}{'CRC failed':<12}Filename\n" + "-"*50 + "\n"

    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write(header)
            f.write(line)
    else:
        with open(summary_file, "a") as f:
            f.write(line)


@app.post("/detect/")
async def detect_watermark(audio: UploadFile = File(...)):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        input_audio_path = os.path.join(OUTPUT_DIR, audio.filename)
        with open(input_audio_path, "wb") as f:
            f.write(audio.file.read())

        base_filename = audio.filename
        if base_filename.startswith("marked_"):
            base_filename = base_filename[len("marked_"):]
        marked_txt_file = os.path.join(OUTPUT_DIR, f"marked_{base_filename}.txt")
        summary_file = os.path.join(OUTPUT_DIR, "detection_summary.txt")

        detect_command = [
            DETECTOR_CLI, "-i", input_audio_path, "--wm-length", "64",
            "--ecc-mode", "1", "--min-freq", "2000", "--max-freq", "8000",
            "--redundancy", "2", "--logfile1", marked_txt_file
        ]
        subprocess.run(detect_command, check=True)

        # Read detected watermark and CRC counts
        detected_message = None
        crc_ok, crc_failed = 0, 0
        with open(marked_txt_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "CRC okay!" in line:
                    detected_message = line.strip()
                elif "CRC correct:" in line:
                    parts = line.strip().split(",")
                    crc_ok = int(parts[0].split(":")[1].strip())
                    crc_failed = int(parts[1].split(":")[1].strip())

        # Update detection summary log
        def update_summary_log(filename: str, ok: int, failed: int, summary_path: str):
            entry = f"{ok:<10}{failed:<12}{filename}\n"
            header = f"{'CRC ok':<10}{'CRC failed':<12}Filename\n" + "-" * 50 + "\n"
            if not os.path.exists(summary_path):
                with open(summary_path, "w") as f:
                    f.write(header)
                    f.write(entry)
            else:
                with open(summary_path, "a") as f:
                    f.write(entry)

        update_summary_log(audio.filename, crc_ok, crc_failed, summary_file)

        if detected_message:
            return {
                "message": "Watermark detected",
                "detected_watermark": detected_message,
                "marked_txt_file": marked_txt_file,
                "summary_log": summary_file
            }
        else:
            os.remove(marked_txt_file)
            raise HTTPException(status_code=404, detail="No watermark detected")

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to detect watermark")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/uploads/")
async def list_uploaded_files():
    files = [f for f in os.listdir(OUTPUT_DIR) if "marked" in f or "detect" in f]
    return {"uploaded_files": files}




