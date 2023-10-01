from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark 
from scipy.io.wavfile import write as write_wav
import subprocess

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/",eval=True)
model.to("cpu")

task = "hello welcome too voice cloning"

output_dict = model.synthesize(
    text,
    config,
    speaker_id="speaker",
    voice_dirs="bark_voices",
    temperature=0.95,
)
write_wav("output.wav",24000,output_dict["wav"])