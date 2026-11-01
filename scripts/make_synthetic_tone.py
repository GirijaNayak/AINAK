"""Create a local PANNs plumbing fixture, not environmental-audio evaluation data."""
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    sample_rate=32000
    time=np.arange(sample_rate*3,dtype=np.float32)/sample_rate
    waveform=(.2*np.sin(2*np.pi*440*time)).astype(np.float32)
    output=Path("artifacts/audio/synthetic_tone.wav")
    output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(output,waveform,sample_rate)
    print(output)


if __name__=="__main__":main()
