

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


dims = [64, 80, 96]
channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
expansion = 2
kernel_size = 3
patch_size = (2, 2)
ih=256
L = [2, 4, 3]
print(3, channels[0], 'kernel=3', 'stride=2')
print(channels[0], channels[1], 1, expansion)
print(channels[1], channels[2], 2, expansion)
print(channels[2], channels[3], 1, expansion)
print(channels[2], channels[3], 1, expansion)
print(channels[3], channels[4], 2, expansion)
print(channels[5], channels[6], 2, expansion)
print(channels[7], channels[8], 2, expansion)
print(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2))
print(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4))
print(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4))
print(channels[-2], channels[-1])
# print('avgpool',ih//2, 1)
# print('Linear=',channels[-1], 'num_classes=',1000, 'bias=False')