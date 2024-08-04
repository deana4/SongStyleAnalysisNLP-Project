import pandas as pd
import spotipy
import yaml
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


config = load_config()

client_credentials_manager = SpotifyClientCredentials(client_id=config['spotify']['client_id'],
                                                      client_secret=config['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def getReleaseYear(artist_name, song_name):
    results = sp.search(q=f'{artist_name} {song_name}', type='track')
    # Extract release year from the first result
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        release_year = track['album']['release_date'][:4]
        return release_year
    else:
        return None


def fillRYears(csvPath):
    data = pd.read_csv(csvPath)
    releaseYears = []
    for row in tqdm(data.itertuples(), total=len(data), desc="Processing rows"):
        artist = getattr(row, 'artist')
        song = getattr(row, 'song')
        retYear = getReleaseYear(artist, song)
        print(song, retYear)
        releaseYears.append(retYear)
        # print(f"artist: {artist}, song: {song}, year: {retYear}")

    # data['ReleaseYear'] = releaseYears
    # data.to_csv(csvPath, index=False, encoding='utf-8-sig')


# Example usage for a Hebrew song
if __name__ == "__main__":
    fillRYears("data.csv")
