##############################################
# DO NOT MODIFY THIS FILE
##############################################


import logging
import torch
import numpy as np
import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from jiwer import wer
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from model import anonymize
import soundfile as sf
import sys
from typing import Union
import librosa
from speechbrain.inference import SpeakerRecognition

import warnings
warnings.simplefilter("ignore", FutureWarning)

# Setup logging to only output to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_total_eer(
    enrollment_dir: Union[str, Path], 
    trial_dir: Union[str, Path], 
    asv_model=None, 
    sr: int = 16000
) -> float:
    """
    Compute Equal Error Rate (EER) between original and anonymized audio files.
    
    Args:
        enrollment_dir (str/Path): Directory containing enrollment audio files
        trial_dir (str/Path): Directory containing trial audio files
        asv_model (Optional): Pre-trained speaker verification model
        sr (int): Sampling rate for audio processing
    
    Returns:
        float: Equal Error Rate (EER)
    """
    # Default logging configuration if not already set
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    # Validate inputs
    enrollment_dir = Path(enrollment_dir)
    trial_dir = Path(trial_dir)
    
    if not enrollment_dir.exists():
        raise FileNotFoundError(f"Enrollment directory not found: {enrollment_dir}")
    
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    
    # Default ASV model if not provided
    if asv_model is None:
        try:
            asv_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
        except ImportError:
            raise ImportError("No ASV model provided and SpeechBrain model could not be imported.")
    
    # Collect file pairs
    enrollment_files = {}
    trial_files = {}
    
    # Find enrollment and trial files
    for speaker_dir in enrollment_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            anon_files = list(speaker_dir.rglob("anon_*.wav"))
            if anon_files:
                enrollment_files[speaker_name] = anon_files[0]
    
    for speaker_dir in trial_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_name = speaker_dir.name
            trial_file = list(speaker_dir.rglob("*.wav"))
            if trial_file:
                trial_files[speaker_name] = trial_file[0]
    
    # Compute similarity scores
    similarity_scores = []
    labels = []
    
    # Process genuine pairs
    for speaker, enroll_file in enrollment_files.items():
        if speaker in trial_files:
            try:
                # Load and process enrollment audio
                y_enroll, _ = librosa.load(enroll_file, sr=sr)
                emb_enroll = asv_model.encode_batch(
                    torch.tensor(y_enroll).unsqueeze(0)
                ).squeeze().numpy()
                
                # Load and process trial audio
                y_trial, _ = librosa.load(trial_files[speaker], sr=sr)
                emb_trial = asv_model.encode_batch(
                    torch.tensor(y_trial).unsqueeze(0)
                ).squeeze().numpy()
                
                # Compute cosine similarity
                score = np.dot(emb_enroll, emb_trial) / (
                    np.linalg.norm(emb_enroll) * np.linalg.norm(emb_trial)
                )
                
                similarity_scores.append(score)
                labels.append(1)  # Genuine pair
            
            except Exception as e:
                logging.warning(f"Error processing genuine pair for {speaker}: {e}")
        
        # Process impostor pairs
        for other_speaker, other_trial_file in trial_files.items():
            if other_speaker != speaker:
                try:
                    # Load and process enrollment audio
                    y_enroll, _ = librosa.load(enroll_file, sr=sr)
                    emb_enroll = asv_model.encode_batch(
                        torch.tensor(y_enroll).unsqueeze(0)
                    ).squeeze().numpy()
                    
                    # Load and process impostor trial audio
                    y_other, _ = librosa.load(other_trial_file, sr=sr)
                    emb_other = asv_model.encode_batch(
                        torch.tensor(y_other).unsqueeze(0)
                    ).squeeze().numpy()
                    
                    # Compute cosine similarity
                    score = np.dot(emb_enroll, emb_other) / (
                        np.linalg.norm(emb_enroll) * np.linalg.norm(emb_other)
                    )
                    
                    similarity_scores.append(score)
                    labels.append(0)  # Impostor pair
                
                except Exception as e:
                    logging.warning(f"Error processing impostor pair: {e}")
    
    # Validate scores
    if len(similarity_scores) == 0:
        raise ValueError("No similarity scores computed. Check audio files and model.")
    
    # Compute EER
    try:
        y_true = np.array(labels)
        y_scores = np.array(similarity_scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Compute EER
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        return eer
    
    except Exception as e:
        logging.error(f"EER computation failed: {e}")
        raise RuntimeError("Could not compute Equal Error Rate.")


def transcribe_audio(audio_path):
    """
    Transcribe audio from a .wav file path.
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        error_msg = f"Audio file not found: {audio_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        transcription = ASR_PIPELINE(str(audio_path))
        return transcription['text'].lower()
    except Exception as e:
        error_msg = f"Error transcribing {audio_path}: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

def compute_we(input_audio_path, anonymized_audio_path):
    """
    Compute Word Error Rate (WE = WER * N) given the input and anonymized audio.
    """
    original = transcribe_audio(input_audio_path)
    anonymized = transcribe_audio(anonymized_audio_path)

    if original is None or anonymized is None:
        error_msg = f"Failed transcription for {input_audio_path}. Stopping evaluation."
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    words = len(original.split())
    we = wer(original, anonymized) * words
    return we, words

def evaluate(evaluation_data_path, anonymization_algorithm):
    """
    Evaluate the anonymization algorithm by computing WER and EER.
    """

    enrollment_directory = os.path.join(evaluation_data_path, "Enrollment")
    trial_directory = os.path.join(evaluation_data_path, "Trial")

    if not os.path.exists(enrollment_directory):
        error_msg = f"Enrollment directory does not exist: {enrollment_directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not os.path.exists(trial_directory):
        error_msg = f"Trial directory does not exist: {trial_directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    total_wer = 0
    total_words = 0
    start = time.time()

    # Process Enrollment and Trial directories
    for subset_directory in [enrollment_directory, trial_directory]:
        subset_name = os.path.basename(subset_directory)  # "Enrollment" or "Trial"

        for speaker in os.listdir(subset_directory):
            speaker_path = os.path.join(subset_directory, speaker)
            if not os.path.isdir(speaker_path):
                continue

            # Ensure the anonymized directory exists
            anonymized_dir = os.path.join(speaker_path, "anonymized")
            os.makedirs(anonymized_dir, exist_ok=True)

            # Collect all .wav files
            audio_files = [
                f for f in os.listdir(speaker_path) 
                if f.lower().endswith('.wav')
            ]

            # If there are no .wav files, log and continue
            if not audio_files:
                logging.warning(f"No audio files found for speaker: {speaker_path}")
                continue

            for filename in tqdm(audio_files, desc=f"Processing {subset_name}/{speaker}"):
                input_audio_path = os.path.join(speaker_path, filename)

                # Ensure file is strictly .wav (reject other formats)
                if not filename.endswith('.wav'):
                    error_msg = f"Invalid file format detected: {filename}. Only .wav files are allowed."
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                # Generate anonymized file path
                anonymized_audio_path = os.path.join(anonymized_dir, f"anon_{filename}")

                try:
                    # Anonymization process
                    anonymized_audio, sr = anonymization_algorithm(input_audio_path)
                    sf.write(anonymized_audio_path, anonymized_audio, sr)
                except Exception as e:
                    error_msg = f"Error anonymizing {filename}: {e}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                try:
                    # Compute WER
                    we, reference_length = compute_we(input_audio_path, anonymized_audio_path)
                    if reference_length == 0:
                        error_msg = f"Reference length is 0 for {filename}. Please ensure the original audio files you are using are not empty and contain english speech. Skipping WER computation."
                        logging.warning(error_msg)
                        continue
                    total_wer += we
                    total_words += reference_length
                except Exception as e:
                    error_msg = f"Error computing WER for {filename}: {e}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)

    eer = compute_total_eer(enrollment_directory, trial_directory)

    end = time.time()

    if total_words == 0:
        error_msg = f"Empty transcriptions. Please ensure the original audio files you are using are not empty and contain english speech."
        logging.error(error_msg)
        raise ValueError(error_msg)

    avg_wer = total_wer / total_words
    results = pd.DataFrame([{"WER": avg_wer, "EER": eer, "Runtime (s)": end - start}])
    results.to_csv("results.csv", index=False)

    logging.info("Evaluation completed successfully. Results saved to results.csv.")

if __name__ == "__main__":
    try:
        evaluation_data_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_data/"
        asr_model_id = sys.argv[2] if len(sys.argv) > 2 else "facebook/wav2vec2-base-960h"
        
        ASR_PIPELINE = pipeline("automatic-speech-recognition", model=asr_model_id, device=device)
        evaluate(evaluation_data_path, anonymize)
    except Exception as e:
        logging.critical(f"Evaluation failed: {e}")
        exit(1)


