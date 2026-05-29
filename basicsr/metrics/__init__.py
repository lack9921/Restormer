from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_lpips



def calculate_metric(data, opt):
    """Calculate metric from data and options."""
    metric_type = opt.pop('type')
    metric_func = globals().get(metric_type)
    if metric_func is None:
        raise ValueError(f'Metric {metric_type} not found.')
    result = metric_func(data.get('img'), data.get('img2'), **opt)
    opt['type'] = metric_type  # restore
    return result

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips']
