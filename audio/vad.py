"""Adaptive energy-based voice activity detection."""


class AdaptiveVAD:
    """Classifies audio frames as speech vs noise using an adaptive RMS floor.

    The first `calibration_frames` frames only build the noise estimate.
    Afterwards a frame is speech when its RMS exceeds
    max(initial_threshold, noise_estimate * multiplier).

    The noise estimate follows the ambient level only during non-speech
    frames, so background noise changes do not require retuning.
    """

    def __init__(self, initial_threshold: float = 0.012, multiplier: float = 3.0,
                 alpha: float = 0.15, calibration_frames: int = 16):
        self.initial_threshold = initial_threshold
        self.multiplier = multiplier
        self.alpha = alpha
        self.calibration_frames = calibration_frames
        self.noise: float = initial_threshold
        self.ready: bool = False
        self._count: int = 0

    def is_speech(self, rms: float) -> bool:
        if not self.ready:
            self.noise += self.alpha * (rms - self.noise)
            self._count += 1
            if self._count >= self.calibration_frames:
                self.ready = True
            return False

        threshold = max(self.initial_threshold, self.noise * self.multiplier)
        speech = rms >= threshold

        if not speech:
            self.noise += self.alpha * (rms - self.noise)

        return speech

    def reset(self) -> None:
        self.noise = self.initial_threshold
        self.ready = False
        self._count = 0