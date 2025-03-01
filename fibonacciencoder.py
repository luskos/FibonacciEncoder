import cv2
import numpy as np
import os
import multiprocessing

def generate_fibonacci(limit=255):
    """Generate Fibonacci numbers up to a given limit."""
    fib = [1, 2]
    while fib[-1] <= limit:
        fib.append(fib[-1] + fib[-2])
    return fib

def fibonacci_encode(n, fib_seq):
    """Encodes a number using Fibonacci representation."""
    result = []
    for i in range(len(fib_seq)-1, 0, -1):
        if fib_seq[i] <= n:
            result.append(fib_seq[i])
            n -= fib_seq[i]
    return sum(result)  # Store only the sum for efficiency

def apply_fibonacci_encoding(image, entropy_map, fib_seq, threshold=3.0):
    """Applies Fibonacci encoding selectively based on entropy."""
    block_size = 16  # 16x16 blocks
    h, w, c = image.shape
    encoded_image = image.copy()
    
    for i in range(0, h - h % block_size, block_size):
        for j in range(0, w - w % block_size, block_size):
            if entropy_map[i // block_size, j // block_size] < threshold:
                for channel in range(c):
                    encoded_image[i:i+block_size, j:j+block_size, channel] = np.vectorize(lambda x: fibonacci_encode(x, fib_seq))(image[i:i+block_size, j:j+block_size, channel])
    
    return encoded_image.astype(np.uint8)  # Ensure proper data type

def compute_entropy(image, block_size=16):
    """Computes entropy map using OpenCV's histogram calculation (optimized)."""
    h, w = image.shape[:2]
    h_blocks, w_blocks = h // block_size, w // block_size
    entropy_map = np.zeros((h_blocks, w_blocks))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = gray_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            hist = cv2.calcHist([block], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            entropy_map[i, j] = -np.sum(hist * np.log2(hist + 1e-8))
    
    return entropy_map

def process_frame(frame, fib_seq, frame_index, total_frames):
    """Processes a single frame using Fibonacci encoding."""
    if frame is None:
        print(f"[ERROR] Frame {frame_index + 1} is None and will be skipped.")
        return None
    
    entropy_map = compute_entropy(frame)
    encoded_frame = apply_fibonacci_encoding(frame, entropy_map, fib_seq)
    
    print(f"[INFO] Processed frame {frame_index + 1}/{total_frames}")
    return encoded_frame

def encode_video(input_video, output_video, frame_skip=3):
    """Encodes a video using Fibonacci hybrid encoding without multiprocessing (for debugging)."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_video}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    fib_seq = generate_fibonacci(255)
    frame_list = []
    
    for i in range(0, total_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARNING] Skipping unreadable frame at index {i}")
            continue
        frame_list.append((frame, fib_seq, i, total_frames))
    
    if not frame_list:
        print("[ERROR] No frames were successfully loaded.")
        cap.release()
        return
    
    encoded_frames = []
    for frame_args in frame_list:
        encoded_frame = process_frame(*frame_args)
        if encoded_frame is not None:
            encoded_frames.append(encoded_frame)
    
    if not encoded_frames:
        print("[ERROR] No frames were processed successfully.")
        return
    
    for encoded_frame in encoded_frames:
        out.write(encoded_frame)
    
    cap.release()
    out.release()
    print(f"[INFO] Encoding Complete: Output saved at {output_video} ({len(encoded_frames)} frames written)")

def compute_psnr(original_video, encoded_video):
    """Computes PSNR between original and encoded video."""
    cap_orig = cv2.VideoCapture(original_video)
    cap_enc = cv2.VideoCapture(encoded_video)
    
    if not cap_enc.isOpened():
        print("[ERROR] Encoded video could not be opened. No PSNR calculation possible.")
        return float('nan')
    
    psnr_values = []
    while cap_orig.isOpened() and cap_enc.isOpened():
        ret1, frame_orig = cap_orig.read()
        ret2, frame_enc = cap_enc.read()
        
        if not ret1 or not ret2:
            break
        
        mse = np.mean((frame_orig - frame_enc) ** 2)
        if mse == 0:
            psnr_values.append(100)
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_values.append(psnr)
    
    cap_orig.release()
    cap_enc.release()
    return np.mean(psnr_values) if psnr_values else float('nan')

if __name__ == "__main__":
    input_video = "C:\\Users\\Lenovo\\Desktop\\mav.mp4"
    output_video = "C:\\Users\\Lenovo\\Desktop\\fibonacci_encoded_video.mp4"
    encode_video(input_video, output_video, frame_skip=3)
    
    psnr = compute_psnr(input_video, output_video)
    print(f"[INFO] Average PSNR: {psnr:.2f} dB")
