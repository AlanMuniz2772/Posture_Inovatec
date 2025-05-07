import os
import pandas as pd
import yt_dlp
import subprocess

# Ruta al CSV
csv_path = "deadlift_videos.csv"

# Carpeta donde guardar recortes
output_dir = "./videos_recortados"
os.makedirs(output_dir, exist_ok=True)

# Cargar dataset
df = pd.read_csv(csv_path)

# Configuración de descarga temporal
temp_dir = "./temp_videos"
os.makedirs(temp_dir, exist_ok=True)

for index, row in df.iterrows():
    video_id = row['youtube_id']
    start = row['time_start']
    end = row['time_end']
    label = row['label']
    split = row['split']

    duration = end - start
    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_output = os.path.join(temp_dir, f"{video_id}.mp4")
    final_output = os.path.join(output_dir, f"{label}_{video_id}_{start}_{end}.mp4")

    # Descargar el video completo solo si no existe
    if not os.path.exists(temp_output):
        print(f"📥 Descargando {video_id}...")
        ydl_opts = {
            'outtmpl': temp_output,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except Exception as e:
                print(f"❌ Error al descargar {video_id}: {e}")
                continue

    # Recortar con ffmpeg
    print(f"✂️ Recortando {video_id} de {start}s a {end}s...")
    command = [
        "ffmpeg",
        "-y",
        "-i", temp_output,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        final_output
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Guardado en: {final_output}")
        os.remove(temp_output)
        print(f"🗑️ Eliminado archivo temporal: {temp_output}")
    except subprocess.CalledProcessError:
        print(f"❌ Error al recortar {video_id}")
