import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale


class ConvolutionInterpolator:
    """Small helper class for building Gaussian kernels and applying
    convolution / interpolation to 2D image arrays.

    No constructor arguments are needed — just instantiate and call the
    methods below.

    Example Usage
    -------------
    >>> ci = ConvolutionInterpolator()
    >>> kernel = ci.gaussian_kernel_2d(sigma=2.5)
    >>> smoothed = ci.convolve(image, kernel)
    >>> small = ci.interpolate(smoothed, scale_factor=0.25)
    """

    def gaussian_kernel_2d(self, sigma):
        """Build a 2D Gaussian kernel from the outer product of a 1D Gaussian.

        The kernel size K is derived from sigma via the rule of thumb that
        a Gaussian's mass is concentrated within about ±3σ of its centre
        (see README) — it is not a separate input.

        Args:
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            ndarray: K x K kernel, dtype float, normalised to sum to 1.
        """
        # Calculate half and K based on the 3-sigma rule
        half = int(np.ceil(3 * sigma))
        K = 2 * half + 1

        # Generate 1D Gaussian samples
        x = np.arange(-half, half + 1, dtype=float)
        g_x = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2))

        # Build 2D kernel using outer product and normalize
        kernel_2d = np.outer(g_x, g_x)
        return kernel_2d / np.sum(kernel_2d)

    def convolve(self, image, kernel, **kwargs):
        """Convolve a 2D image with a kernel.

        Args:
            image  (ndarray): H x W greyscale image.
            kernel (ndarray): Q x Q filter kernel, e.g. from gaussian_kernel_2d.
            **kwargs:
                mode      (str): Output size passed to scipy.signal.convolve2d
                                  — 'same', 'valid', or 'full' (default 'same').
                boundary  (str): Boundary/padding style — 'fill', 'wrap', or
                                  'symm' (default 'symm').
                fillvalue (float): Constant used when boundary='fill'
                                    (default 0).

        Returns:
            ndarray: Filtered image — same shape as the input under the
            default mode='same', or a different shape for 'valid'/'full'.
        """
        mode = kwargs.pop('mode', 'same')
        boundary = kwargs.pop('boundary', 'symm')
        fillvalue = kwargs.pop('fillvalue', 0)

        # using parameters like mode, boundary and fillvalue in covolve2d  function
        return convolve2d(image, kernel, mode=mode, boundary=boundary, fillvalue=fillvalue)

    def interpolate(self, image, **kwargs):
        """Resample a 2D image by a given scale factor.

        Args:
            image (ndarray): H x W greyscale image.
            **kwargs:
                scale_factor (float): Output size relative to input, e.g.
                                       0.25 downsamples to a quarter size
                                       (default 0.5).
                order        (int):   Interpolation order — 0 = nearest
                                       neighbour, 1 = bilinear, 3 = bicubic,
                                       etc. (default 0).

        Returns:
            ndarray: Resampled image, shape scaled by scale_factor.
        """
        #creating vars - scale factor and order with default values 
        scale_factor = kwargs.pop('scale_factor', 0.5)
        order = kwargs.pop('order', 0)

        # Use skimage.transform.rescale with anti_aliasing=False and mode='reflect'
        return rescale(image, scale_factor, order=order, anti_aliasing=False, mode='reflect')
