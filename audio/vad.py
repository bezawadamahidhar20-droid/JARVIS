"""Adaptive energy-based voice activity detection.

All public methods and attributes are fully type-annotated for mypy / pyright
compatibility.
"""


class AdaptiveVAD:
    """Classifies audio frames as speech vs noise using an adaptive RMS floor.

    Algorithm
    ---------
    1. The first ``calibration_frames`` frames **only** update the noise
       estimate; ``is_speech()`` always returns ``False`` during this phase.
    2. After calibration, a frame is classified as *speech* when its RMS
       exceeds ``max(initial_threshold, noise_estimate × multiplier)``.
    3. The noise estimate is updated **only** during non-speech frames using
       exponential moving average (EMA), so sudden bursts of speech do not
       corrupt the baseline.

    Parameters
    ----------
    initial_threshold:
        Absolute RMS floor below which nothing is ever considered speech.
        Guards against spurious VAD triggers in near-silence.
    multiplier:
        A frame is speech when ``rms >= noise_estimate × multiplier``.
        Higher values require louder speech relative to ambient noise.
    alpha:
        EMA smoothing factor for the noise estimate (0 < alpha < 1).
        Smaller values make the estimate react more slowly to changes.
    calibration_frames:
        Number of frames used to build the initial noise estimate before
        speech detection begins.
    """

    def __init__(
        self,
        initial_threshold: float = 0.012,
        multiplier: float = 3.0,
        alpha: float = 0.15,
        calibration_frames: int = 16,
    ) -> None:
        self.initial_threshold: float = initial_threshold
        self.multiplier: float = multiplier
        self.alpha: float = alpha
        self.calibration_frames: int = calibration_frames

        # Runtime state — reset via reset().
        self.noise: float = initial_threshold
        self.ready: bool = False
        self._count: int = 0

    def is_speech(self, rms: float) -> bool:
        """Classify one audio frame.

        Parameters
        ----------
        rms:
            Root-mean-square energy of the current audio frame (non-negative).

        Returns
        -------
        bool
            ``True`` if the frame is classified as speech, ``False`` otherwise.
            Always returns ``False`` during the calibration phase.
        """
        if not self.ready:
            # Calibration phase: update noise estimate, do not classify.
            self.noise += self.alpha * (rms - self.noise)
            self._count += 1
            if self._count >= self.calibration_frames:
                self.ready = True
            return False

        threshold: float = max(self.initial_threshold, self.noise * self.multiplier)
        speech: bool = rms >= threshold

        if not speech:
            # Only update noise during silence so speech energy cannot raise
            # the baseline and mask quieter utterances.
            self.noise += self.alpha * (rms - self.noise)

        return speech

    def reset(self) -> None:
        """Reset the VAD to its initial uncalibrated state.

        Call this before each new capture session so ambient-noise estimates
        from previous sessions do not carry over.
        """
        self.noise = self.initial_threshold
        self.ready = False
        self._count = 0
