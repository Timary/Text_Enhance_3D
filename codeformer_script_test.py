import subprocess
import sys
from pathlib import Path

def run_codeformer_inference(input_path='test', weight=0.5, has_aligned=True):
    """
    Runs the inference_codeformer.py script with specified parameters.
    
    Args:
        input_path (str): Path to the input directory
        weight (float): Weight parameter for CodeFormer
        has_aligned (bool): Whether faces are already aligned
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    try:
        # Build the command
        cmd = [
            sys.executable,
            'inference_codeformer.py',
            '-w', str(weight),
            '--input_path', input_path
        ]
        
        # Add has_aligned flag if True
        if has_aligned:
            cmd.append('--has_aligned')
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        return result.returncode, result.stdout, result.stderr
        
    except Exception as e:
        return -1, "", str(e)

def run_inference_inpainting(input_path='inputs/masked_faces'):
    """
    Runs the inference_inpainting.py script with the specified input path.
    
    Args:
        input_path (str): Path to the input directory containing masked faces
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    try:
        # Build the command
        cmd = [
            sys.executable,
            'inference_inpainting.py',
            '--input_path', input_path
        ]
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        return result.returncode, result.stdout, result.stderr
        
    except Exception as e:
        return -1, "", str(e)

# Usage examples:
if __name__ == "__main__":
    # Run CodeFormer inference
    print("Running CodeFormer inference...")
    return_code, output, error = run_codeformer_inference('test', 0.5, True)
    if return_code == 0:
        print("CodeFormer Success:", output)
    else:
        print("CodeFormer Error:", error)
    
    # Run inpainting inference
    print("\nRunning inpainting inference...")
    return_code, output, error = run_inference_inpainting('inputs/masked_faces')
    if return_code == 0:
        print("Inpainting Success:", output)
    else:
        print("Inpainting Error:", error)