from calendar import c
import torch

from torch.nn import Module


class StftArray(Module):
    def __init__(self, n_dft=1024, hop_size=512, window_length=None,
                 onesided=True, is_complex=True, complex_as_channels=False,
                 mag_only=False, phase_only=False, real_only=False):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.onesided = onesided
        self.is_complex = is_complex
        self.complex_as_channels = complex_as_channels

        self.mag_only = mag_only
        self.phase_only = phase_only
        self.real_only = real_only
        self.window_length = n_dft if window_length is None else window_length
        
    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        input_shape = x.shape

        if len(input_shape) == 3:
            # (batch_size, n_channels, time_steps) => Microphone array
            # Collapse channels into batch
            x = x.flatten(end_dim=1)


        window = torch.hann_window(self.window_length, device=x.device)
        y = torch.stft(x, self.n_dft, hop_length=self.hop_size, 
                       onesided=self.onesided, return_complex=True,
                       win_length=self.window_length, window=window)
        y = y[:, 1:] # Remove DC component (f=0hz)

        y = y.transpose(1, 2)
        # y.shape == (batch_size*channels, time, freqs)

        if len(input_shape) == 3:
            batch_size, num_channels, _ = input_shape
            # De-collapse first dim into batch and channels
            y = y.unflatten(0, (batch_size, num_channels))

        if self.mag_only:
            return y.abs()
        if self.phase_only:
            return y.angle()
        if self.real_only:
            return y.real

        if not self.is_complex:
            y = _complex_to_real(y, self.complex_as_channels)

        return y


class StftPhaseArray(StftArray):
    def __init__(self, features):
        super().__init__(features["n_dft"],
                         features["hop_size"],
                         features["onesided"],
                         phase_only=True)

        self.n_output_channels = 2


class CrossSpectra(StftArray):
    def __init__(self, n_dft=1024, hop_size=512,
                 onesided=True, is_complex=True, complex_as_channels=False,
                 phase_only=False):        
        super().__init__(n_dft, hop_size, onesided=onesided)

        self._is_complex = is_complex # _ is added not to conflict with StftArray
        self.complex_as_channels = complex_as_channels
        self.phase_only = phase_only

    def forward(self, X):
        "Expected input has shape (batch_size, n_channels, time_steps)"
        batch_size, n_channels, time_steps = X.shape

        stfts = super().forward(X)
        # (batch_size, n_channels, n_time_bins, n_freq_bins)
        y = []

        # Compute the cross-spectrum between each pair of channels
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                y_ij = stfts[:, i]*stfts[:, j].conj()
                y.append(y_ij)
        
        y = torch.stack(y, dim=1)

        if self.phase_only:
            return y.angle()

        if not self._is_complex:
            _complex_to_real(y, self.complex_as_channels)

        return y


def _complex_to_real(x, as_channels=False): 
    y = torch.view_as_real(x)
    if as_channels:
        # Merge channels and real and imaginary parts (last dim) as separate input channels
        y = y.transpose(2, -1).flatten(start_dim=1, end_dim=2)

    return y
