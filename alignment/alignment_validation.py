import numpy as np
import torch
import pickle
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torchvision import transforms

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import re
import os
import gc

from utils import image_normalization
from alignment.alignment_model import *
from alignment.alignment_model import _LinearAlignment, _MLPAlignment, _ConvolutionalAlignment, _TwoConvAlignment, _ZeroShotAlignment
from alignment.alignment_utils import *
from alignment.alignment_training import *
from alignment.alignment_validation import *
from utils import get_psnr


def validation_worker(model, inputs, gt, times, worker_id):
    """Worker function for parallel model inference"""
    model.eval()
    psnr_sum = torch.zeros(inputs.shape[0], device=inputs.device)
    
    with torch.no_grad():
        for _ in range(times):
            demo_image = model(inputs)
            demo_image = image_normalization('denormalization')(demo_image)
            psnr_sum += get_batch_psnr(demo_image, gt)
    
    return psnr_sum / times


def validation_parallel_inference(model, dataloader, times, device, num_workers=None):
    """Version with parallel inference for multiple runs"""
    model = model.to(device)
    model.eval()
    
    # Auto-detect optimal number of workers
    if num_workers is None:
        num_workers = min(times, torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count())
    
    total_psnr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, *_ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            
            # Denormalize ground truth once
            gt = image_normalization('denormalization')(inputs)
            
            if times == 1:
                # No need for parallelization with single run
                demo_image = model(inputs)
                demo_image = image_normalization('denormalization')(demo_image)
                batch_psnr = get_batch_psnr(demo_image, gt).sum().item()
            else:
                # Parallel inference for multiple runs
                runs_per_worker = times // num_workers
                remaining_runs = times % num_workers
                
                batch_psnr_sum = torch.zeros(batch_size, device=device)
                
                # Use thread pool for GPU parallelization (better for CUDA)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    
                    # Submit jobs with different number of runs per worker
                    for i in range(num_workers):
                        worker_runs = runs_per_worker + (1 if i < remaining_runs else 0)
                        if worker_runs > 0:
                            future = executor.submit(
                                validation_worker, 
                                model, inputs, gt, worker_runs, i
                            )
                            futures.append(future)
                    
                    # Collect results
                    for future in futures:
                        batch_psnr_sum += future.result()
                
                batch_psnr = batch_psnr_sum.sum().item()
            
            total_psnr += batch_psnr
            total_samples += batch_size
    
    return total_psnr / total_samples


def validation_parallel_batches(model, dataloader, times, device, num_workers=4):
    """Version with parallel batch processing"""
    model = model.to(device)
    model.eval()
    
    def process_batch(batch_data):
        inputs, *_ = batch_data
        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        
        # Denormalize ground truth once
        gt = image_normalization('denormalization')(inputs)
        
        # Accumulate PSNR across multiple runs
        batch_psnr_sum = torch.zeros(batch_size, device=device)
        
        with torch.no_grad():
            for _ in range(times):
                demo_image = model(inputs)
                demo_image = image_normalization('denormalization')(demo_image)
                batch_psnr_sum += get_batch_psnr(demo_image, gt)
        
        batch_mean_psnr = (batch_psnr_sum / times).sum().item()
        return batch_mean_psnr, batch_size
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_batch, dataloader))
    
    total_psnr = sum(psnr for psnr, _ in results)
    total_samples = sum(samples for _, samples in results)
    
    return total_psnr / total_samples


def validation_vectorized(model, dataloader, times, device):
    """Vectorized version for maximum efficiency when memory allows"""
    model = model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, *_ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            
            # Denormalize ground truth once
            gt = image_normalization('denormalization')(inputs)
            
            if times == 1:
                demo_image = model(inputs)
                demo_image = image_normalization('denormalization')(demo_image)
                batch_psnr = get_batch_psnr(demo_image, gt).sum().item()
            else:
                # Vectorized computation - process all runs at once
                # Repeat inputs for all runs
                inputs_repeated = inputs.repeat(times, 1, 1, 1)
                
                # Single forward pass for all runs
                demo_images = model(inputs_repeated)
                demo_images = image_normalization('denormalization')(demo_images)
                
                # Reshape to separate runs and batch dimension
                demo_images = demo_images.view(times, batch_size, *demo_images.shape[1:])
                gt_repeated = gt.unsqueeze(0).repeat(times, 1, 1, 1, 1)
                
                # Compute PSNR for all runs at once
                psnr_all_runs = torch.stack([
                    get_batch_psnr(demo_images[i], gt_repeated[i]) 
                    for i in range(times)
                ])
                
                # Average across runs and sum across batch
                batch_psnr = psnr_all_runs.mean(dim=0).sum().item()
            
            total_psnr += batch_psnr
            total_samples += batch_size
    
    return total_psnr / total_samples


def validation(model, dataloader, times, device, method='auto', num_workers=None):
    """
    Optimized validation with multiple parallelization strategies
    
    Args:
        model: The model to validate
        dataloader: Data loader for validation data
        times: Number of inference runs per sample
        method: 'auto', 'vectorized', 'parallel_inference', 'parallel_batches', or 'sequential'
        num_workers: Number of parallel workers (auto-detected if None)
    """
    
    if method == 'auto':
        # Auto-select best method based on conditions
        if times == 1:
            method = 'sequential'
        elif times <= 4 and torch.cuda.is_available():
            method = 'vectorized'  # Best for GPU with moderate times
        elif times > 4:
            method = 'parallel_inference'  # Best for many inference runs
        else:
            method = 'parallel_batches'  # Best for CPU or complex cases
    
    if method == 'vectorized':
        return validation_vectorized(model, dataloader, times)
    elif method == 'parallel_inference':
        return validation_parallel_inference(model, dataloader, times, num_workers)
    elif method == 'parallel_batches':
        return validation_parallel_batches(model, dataloader, times, num_workers)
    else:  # sequential
        return validation_sequential(model, dataloader, times)


def validation_sequential(model, dataloader, times, device):
    """Original optimized sequential version for comparison"""
    model = model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, *_ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            
            gt = image_normalization('denormalization')(inputs)
            batch_psnr_sum = torch.zeros(batch_size, device=device)
            
            for _ in range(times):
                demo_image = model(inputs)
                demo_image = image_normalization('denormalization')(demo_image)
                batch_psnr_sum += get_batch_psnr(demo_image, gt)
            
            batch_mean_psnr = (batch_psnr_sum / times).sum().item()
            total_psnr += batch_mean_psnr
            total_samples += batch_size
    
    return total_psnr / total_samples


def prepare_image(image_path, resolution):
    if resolution is None:
        transform = transforms.Compose([transforms.ToTensor(), ])

    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((resolution, resolution))])

    test_image = Image.open(image_path)
    test_image.load()
    test_image = transform(test_image)

    return test_image

def prepare_aligner(aligner_fp, device, resolution, c, n_samples=None):
    if "linear" in aligner_fp or "neural" in aligner_fp:
        aligner = _LinearAlignment(resolution**2)
    
    elif "mlp" in aligner_fp:
        aligner = _MLPAlignment(input_dim=resolution**2, hidden_dims=[resolution**2])

    elif "twoconv" in aligner_fp:
        aligner = _TwoConvAlignment(in_channels=2*c, hidden_channels=2*c, out_channels=2*c, kernel_size=5)

    elif "conv" in aligner_fp:
        aligner = _ConvolutionalAlignment(in_channels=2*c, out_channels=2*c, kernel_size=5)
    
    elif "zeroshot" in aligner_fp:
        if n_samples is None:
            filename = os.path.basename(aligner_fp)
            match = re.search(r'_(\d+)\.pth$', filename)
            n_samples = int(match.group(1))

        aligner = _ZeroShotAlignment(
            F_tilde=torch.zeros(n_samples, resolution**2),
            G_tilde=torch.zeros(resolution**2, n_samples), 
            G=torch.zeros(1, 1),
            L=torch.zeros(n_samples, n_samples),
            mean=torch.zeros(n_samples, 1)
        )

    aligner.load_state_dict(torch.load(aligner_fp, map_location=device))

    return aligner


def prepare_models(model1_fp, model2_fp, aligner_fp, snr, c, resolution, channel_type, device):
    model1 = load_deep_jscc(model1_fp, snr, c, channel_type)
    model2 = load_deep_jscc(model2_fp, snr, c, channel_type)

    encoder = copy.deepcopy(model1.encoder)
    decoder = copy.deepcopy(model2.decoder)

    aligner = prepare_aligner(aligner_fp, device, resolution, c)

    aligned_model = AlignedDeepJSCC(encoder, decoder, aligner, snr, channel_type)
    unaligned_model = AlignedDeepJSCC(encoder, decoder, None, snr, channel_type)
    no_mismatch_model = AlignedDeepJSCC(encoder, copy.deepcopy(load_deep_jscc(model1_fp, snr, c, channel_type).decoder), None, snr, channel_type)

    return no_mismatch_model, unaligned_model, aligned_model


def get_image_aligner(aligned_model, image_path, output_path, resolution, upscale_factor):
    aligned_model = aligned_model.to("cpu")

    test_image = prepare_image(image_path, resolution)

    demo_image = aligned_model(test_image)
    demo_image = image_normalization('denormalization')(demo_image)
    demo_image = image_normalization('normalization')(demo_image)

    demo_image = demo_image.squeeze()
    demo_image = demo_image.cpu().detach().numpy()
    demo_image = demo_image.transpose(1, 2, 0)

    # convert to PIL image and upscale
    pil_image = Image.fromarray((demo_image * 255).astype(np.uint8))
    new_size = (pil_image.width * upscale_factor, pil_image.height * upscale_factor)
    pil_image = pil_image.resize(new_size, Image.NEAREST)  # Use NEAREST or BICUBIC

    pil_image.save(output_path)


def visualization_pipeline(model, image_path, resolution, times, upscale_factor):  
    test_image = prepare_image(image_path, resolution)

    psnr_all = 0.0

    model.to("cpu")

    with torch.no_grad():
        for _ in range(times):
            demo_image = model(test_image)
            demo_image = image_normalization('denormalization')(demo_image)
            gt = image_normalization('denormalization')(test_image)
            psnr_all += get_psnr(demo_image, gt)

        # prepare image for visualization
        demo_image = image_normalization('normalization')(demo_image)
        demo_image = torch.cat([test_image, demo_image.squeeze()], dim=1)  # (C, H, W)
        demo_image = demo_image.numpy()  # (C, H, W)
        demo_image = demo_image.transpose(1, 2, 0)  # convert to (H, W, C) for PIL

        # convert to PIL image and upscale
        pil_image = Image.fromarray((demo_image * 255).astype(np.uint8))
        new_size = (pil_image.width * upscale_factor, pil_image.height * upscale_factor)
        pil_image = pil_image.resize(new_size, Image.NEAREST)  # Use NEAREST or BICUBIC

    # show the upscaled image
    plt.figure(figsize=(new_size[0] / 100, new_size[1] / 100), dpi=100)
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()

    print("Average PSNR is {:.2f} over {} runs.".format(psnr_all.item() / times, times))