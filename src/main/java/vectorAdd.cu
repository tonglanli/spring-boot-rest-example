extern "C"
__global__ void add(int n, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + y[i];
}