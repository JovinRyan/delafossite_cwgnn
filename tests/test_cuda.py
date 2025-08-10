import torch
import datetime
import sys

def get_device_info():
    lines = []
    lines.append("="*40)
    lines.append("PyTorch CUDA Diagnostics")
    lines.append(f"Timestamp: {datetime.datetime.now()}")
    lines.append("="*40)

    try:
        lines.append(f"PyTorch Version: {torch.__version__}")
        lines.append(f"CUDA Built: {torch.backends.cuda.is_built()}")
        lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        lines.append(f"torch.version.cuda: {torch.version.cuda}")
        lines.append(f"Number of CUDA Devices: {torch.cuda.device_count()}")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                lines.append(f"\n--- Device {i} ---")
                lines.append(f"Name: {torch.cuda.get_device_name(i)}")
                lines.append(f"Capability: {torch.cuda.get_device_capability(i)}")
                lines.append(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                lines.append(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        else:
            lines.append("CUDA is NOT available. Please check your drivers or PyTorch installation.")

    except Exception as e:
        lines.append("\nAn error occurred during CUDA diagnostics:")
        lines.append(str(e))

    return "\n".join(lines)


if __name__ == "__main__":
    output = get_device_info()

    # Print to terminal
    print(output)

    # Write to file
    with open("test_cuda_log.txt", "w") as f:
        f.write(output + "\n")
