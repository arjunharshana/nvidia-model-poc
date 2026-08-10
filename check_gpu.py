import torch, time

print("CUDA available:", torch.cuda.is_available())
print("Device name :", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)

size = 4096  # bigger matrix makes the Tensor Core advantage more visible
a = torch.randn(size, size, device="cuda")
b = torch.randn(size, size, device="cuda")

def bench(dtype, warmup=10, iters=50):
    a_, b_ = a.to(dtype), b.to(dtype)
    # warm-up: not timed, lets CUDA pick/cache the right kernels
    for _ in range(warmup):
        c = a_ @ b_
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        c = a_ @ b_
    torch.cuda.synchronize()
    return (time.time() - start) / iters

fp32_time = bench(torch.float32)
fp16_time = bench(torch.float16)
print(f"FP32 matmul avg: {fp32_time*1000:.3f} ms")
print(f"FP16 matmul avg: {fp16_time*1000:.3f} ms  (Tensor Cores)")
print(f"Speedup: {fp32_time/fp16_time:.2f}x")