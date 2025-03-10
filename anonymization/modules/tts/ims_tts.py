import torch
import resampy

from .AnonFastSpeech2_multi import AnonFastSpeech2
from utils import setup_logger

logger = setup_logger(__name__)

class ImsTTS:

    def __init__(self, hifigan_path, fastspeech_path, device, embedding_path=None, output_sr=16000, lang='en'):
        self.device = device
        self.output_sr = output_sr

        self.model = AnonFastSpeech2(device=self.device, vocoder_model_path=hifigan_path,
                                     tts_model_path=fastspeech_path, embedding_model_path=embedding_path,
                                     language=lang, faster_vocoder=True)

    def read_text(self, text, speaker_embedding, text_is_phones=True, duration=None, pitch=None, energy=None,
                  start_silence=None, end_silence=None):

        self.model.default_utterance_embedding = speaker_embedding.to(self.device)
        try:
            wav = self.model(text=text, text_is_phonemes=text_is_phones, durations=duration, pitch=pitch, energy=energy)
        except IndexError:
            logger.info(f'Index Error for utterance {text}')
            return None
        except RuntimeError:
            logger.info(f'Runtime Error for utterance {text}')
            return None

        # TODO: this is not an ideal solution, but must work for now
        i = 0
        while wav.shape[0] < 12000:  # 0.5 s
            # sometimes, the speaker embedding is so off that it leads to a practically empty audio
            # then, we need to sample a new embedding
            if i > 0 and i % 10 == 0:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
            else:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(self.device)
            speaker_embedding = speaker_embedding * mask
            self.model.default_utterance_embedding = speaker_embedding.to(self.device)
            wav = self.model(text=text, text_is_phonemes=text_is_phones, durations=duration, pitch=pitch, energy=energy)
            i += 1
            if i > 30:
                break
        if i > 0:
            logger.info(f'Synthesized utt in {i} takes')

        # start and end silence are computed for 16000, so we have to adapt this to different output sr
        factor = self.output_sr // 16000
        if start_silence is not None:
            start_sil = torch.zeros([start_silence * factor]).to(self.device)
            wav = torch.cat((start_sil, wav), dim=0)
        if end_silence is not None:
            end_sil = torch.zeros([end_silence * factor]).to(self.device)
            wav = torch.cat((wav, end_sil), dim=0)

        if self.output_sr != 24000:
            wav = resampy.resample(wav.cpu().numpy(), 24000, self.output_sr)
        else:
            wav = wav.cpu().numpy()

        return wav
