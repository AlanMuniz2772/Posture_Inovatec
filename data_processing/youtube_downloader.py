import yt_dlp
import os

def descargar_video(url, carpeta_destino="./videos"):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    opciones = {
        'outtmpl': os.path.join(carpeta_destino, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(opciones) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    descargar_video("https://www.youtube.com/watch?v=0XL4cZR2Ink")
