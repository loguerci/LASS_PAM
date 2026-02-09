Pour le setup le code, il faut creer un environnement en utilisant le fichier "environment.yml", 
Version python : 3.7.10
Il faudra pip install (en plus) : laion-clap

Ensuite il faut les fichiers ckpt.pt (modele sauvegardé) :
- LASSNet.pt à mettre dans un dossier ckpt (que vous devez créer) : https://drive.google.com/file/d/1f8eCCYYaBdhsFqoi7PJMrT9Oo7GaWLdR/view?usp=sharing
- Les .pt pour CLAP :
    - Modele "music" : https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
    - Modele "music & speech" : https://huggingface.co/lukewys/laion_clap/blob/main/music_speech_epoch_15_esc_89.25.pt

Il faut aussi le dataset pour le training :
- BabySlakh (pour tester) : https://zenodo.org/records/4603870
- Slakh en 16kHz (dataset de 145 Go) : https://zenodo.org/records/7717249

voici ma pip list pour verifier vos intallations :
Package                       Version
----------------------------- -------------------
absl-py                       1.0.0
appdirs                       1.4.4
audioread                     2.1.9
backcall                      0.2.0
backports.functools-lru-cache 1.6.4
braceexpand                   0.1.7
cached-property               1.5.2
cachetools                    5.0.0
certifi                       2021.10.8
cffi                          1.14.5
chardet                       4.0.0
click                         8.0.3
coverage                      6.2
cycler                        0.10.0
dataclasses                   0.8
debugpy                       1.5.1
decorator                     4.4.2
docker-pycreds                0.4.0
entrypoints                   0.4
exceptiongroup                1.3.1
fastrlock                     0.8.3
filelock                      3.4.0
fsspec                        2023.1.0
ftfy                          6.1.1
future                        0.18.2
gitdb                         4.0.12
gitpython                     3.1.46
google-auth                   2.6.0
google-auth-oauthlib          0.4.6
grpcio                        1.43.0
h5py                          3.8.0
huggingface-hub               0.16.4
idna                          2.10
imageio                       2.9.0
importlib-metadata            4.8.2
iniconfig                     2.0.0
ipykernel                     6.7.0
ipython                       7.31.1
jedi                          0.18.1
joblib                        1.0.1
jupyter-client                7.1.2
jupyter-core                  4.9.1
kiwisolver                    1.3.1
laion-clap                    1.1.5
librosa                       0.8.1
llvmlite                      0.31.0
Markdown                      3.3.6
matplotlib                    3.4.2
matplotlib-inline             0.1.3
mkl-fft                       1.3.0
mkl-random                    1.2.1
mkl-service                   2.3.0
nest-asyncio                  1.5.4
networkx                      2.5.1
numba                         0.48.0
numpy                         1.21.6
oauthlib                      3.2.0
olefile                       0.46
packaging                     20.9
pandas                        1.3.5
parso                         0.8.3
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        8.2.0
pip                           21.0.1
platformdirs                  4.0.0
pluggy                        1.2.0
pooch                         1.3.0
progressbar                   2.5
prompt-toolkit                3.0.26
protobuf                      3.19.4
psutil                        7.2.2
ptyprocess                    0.7.0
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycparser                     2.20
Pygments                      2.11.2
pyparsing                     2.4.7
pytest                        7.4.4
python-dateutil               2.8.1
pytz                          2021.3
PyWavelets                    1.1.1
PyYAML                        5.4.1
pyzmq                         19.0.2
regex                         2021.11.10
requests                      2.25.1
requests-oauthlib             1.3.1
resampy                       0.2.2
rsa                           4.8
sacremoses                    0.0.46
safetensors                   0.4.5
scikit-image                  0.18.1
scikit-learn                  0.24.2
scipy                         1.6.3
sentencepiece                 0.1.96
sentry-sdk                    2.51.0
setproctitle                  1.3.3
setuptools                    52.0.0.post20210125
six                           1.15.0
smart-open                    5.0.0
smmap                         5.0.2
SoundFile                     0.10.3.post1
tensorboard                   2.8.0
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.1
tensorboardX                  2.4.1
threadpoolctl                 2.1.0
tifffile                      2021.4.8
tokenizers                    0.13.3
tomli                         2.0.1
torch                         1.8.1
torch-tb-profiler             0.3.1
torchaudio                    0.8.0a0+e4e171a
torchlibrosa                  0.0.9
torchvision                   0.9.1
tornado                       6.1
tqdm                          4.60.0
traitlets                     5.1.1
transformers                  4.30.2
typing                        3.7.4.3
typing-extensions             4.7.1
urllib3                       1.26.20
wandb                         0.18.7
wcwidth                       0.2.5
webdataset                    0.2.100
Werkzeug                      2.0.2
wget                          3.2
wheel                         0.36.2
zipp                          3.6.0