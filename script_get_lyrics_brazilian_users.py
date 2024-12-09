import json
import time
import pandas as pd
from lyricsgenius import Genius
import argparse
from datetime import datetime

def extract_first_line(text):
    first_line = text.splitlines()[0] if text else ""
    return first_line.replace(" ", "").lower()

def fetch_lyrics(part, total_parts):
    lyrics = {}
    no_results_tracks = set()
    tracks = set()
    start_time = time.time()

    # Configurar Genius API
    token = "F0KAQNr8vWCL2MSN7BVsTblWV70K5CcyA6RSZzerlTdOtUHJY2S7Bdn0e3W9vu1q"
    genius = Genius(access_token=token, timeout=60, sleep_time=1)
    genius.remove_section_headers = True

    # Carregar CSV de eventos de escuta
    input_file = 'listening_events_brazil.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Arquivo {input_file} não encontrado. Verifique o nome e tente novamente.")
        return

    # Verificar progresso existente e remover músicas já processadas ou não encontradas
    processed_track_ids = set()
    no_results_track_ids = set()

    for i in range(1, total_parts + 1):
        progress_file = f'lyrics_progress_part_{i}.csv'
        no_results_file = f'no_results_tracks_part_{i}.csv'

        if pd.io.common.file_exists(progress_file):
            try:
                progress_df = pd.read_csv(progress_file)
                processed_track_ids.update(progress_df['trackid'])
                
                lyrics.update(dict(zip(progress_df['trackid'], progress_df['lyric'])))
            except Exception as e:
                print(f"Erro ao carregar progresso existente da parte {i}: {e}")

        if pd.io.common.file_exists(no_results_file):
            try:
                no_results_df = pd.read_csv(no_results_file)
                no_results_track_ids.update(no_results_df['trackid'])
            except Exception as e:
                print(f"Erro ao carregar músicas não encontradas da parte {i}: {e}")
                
    df = df[['trackname', 'artistname', 'trackid']].drop_duplicates(subset='trackid').reset_index(drop=True)
    total_tracks = len(df)
    tracks_per_part = total_tracks // total_parts
    start_index = (part - 1) * tracks_per_part
    end_index = start_index + tracks_per_part if part < total_parts else total_tracks

    # Selecionar apenas as faixas da parte atual
    df = df.iloc[start_index:end_index]

    # Remover músicas já processadas e não encontradas
    df = df[~df['trackid'].isin(processed_track_ids | no_results_track_ids)]
    print(f"Removidas {len(processed_track_ids)} músicas já processadas e {len(no_results_track_ids)} músicas não encontradas do corpus de um total de {total_tracks}.")

    print(f"Iniciando processamento de {len(df)} musicas na parte {part}/{total_parts}.")
    tracks = processed_track_ids | no_results_track_ids
    # Iterar sobre as faixas na fração atribuída
    for index, row in df.iterrows():
        trackname = row['trackname']
        artistname = row['artistname']
        trackid = row['trackid']

        # Evitar duplicatas
        if trackid in tracks:
            print(f"Track: {trackname}, Artist: {artistname} já foi processada")
            continue
        tracks.add(trackid)

        print(f"Track: {trackname}, Artist: {artistname}")

        # Buscar letras no Genius
        try:
            song = genius.search_song(artist=artistname, title=trackname)
            first_line = extract_first_line(song.lyrics) if song else ""
            trackname_edit = trackname.replace(" ", "").lower()
            if song:
                if trackname_edit not in first_line:
                    print("not found")
                    no_results_track_ids.add(trackid)
                    continue
                lyrics[trackid] = song.lyrics
            else:
                print("not found")
                no_results_track_ids.add(trackid)
        except Exception as e:
            print(f"Erro ao buscar {trackname} - {artistname}: {e}")
            continue

        # Salvar progresso a cada 1 minuto
        if time.time() - start_time >= 60:
            lyrics_df = pd.DataFrame([[key, value] for key, value in lyrics.items()], columns=["trackid", "lyric"])
            lyrics_df.to_csv(f'lyrics_progress_part_{part}.csv', index=False)

            no_results_df = pd.DataFrame(list(no_results_track_ids), columns=["trackid"])
            no_results_df.to_csv(f'no_results_tracks_part_{part}.csv', index=False)

            print(f"Progresso salvo em: lyrics_progress_part_{part}.csv e no_results_tracks_part_{part}.csv")
            start_time = time.time()  # Resetar o timer

    # Salvar todas as letras da fração ao final
    output_lyrics_file = f'lyrics_part_{part}.csv'

    # Letras encontradas
    lyrics_df = pd.DataFrame([[key, value] for key, value in lyrics.items()], columns=["trackid", "lyric"])
    lyrics_df.to_csv(output_lyrics_file, index=False)
    print(f"Letras salvas em: {output_lyrics_file}")

    # Faixas sem resultados
    no_results_df = pd.DataFrame(list(no_results_track_ids), columns=["trackid"])
    no_results_df.to_csv(f'no_results_tracks_part_{part}.csv', index=False)
    print(f"Faixas sem resultados salvas em: no_results_tracks_part_{part}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar fração do corpus para buscar letras de músicas.")
    parser.add_argument("part", type=int, help="Índice da parte a ser processada (ex: 1 para a primeira parte).")
    parser.add_argument("total_parts", type=int, help="Número total de partes em que o corpus será dividido.")
    args = parser.parse_args()

    if args.part < 1 or args.part > args.total_parts:
        print("Erro: o índice da parte deve estar entre 1 e o número total de partes.")
    else:
        fetch_lyrics(args.part, args.total_parts)
