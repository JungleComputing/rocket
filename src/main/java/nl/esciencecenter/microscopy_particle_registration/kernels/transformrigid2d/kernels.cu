


extern "C"
__global__ void
TransformRigid2D(const double *input, double *output, int n, double cos_r, double sin_r, double tx, double ty) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    double ix = input[2 * i + 0];
    double iy = input[2 * i + 1];

    double ox = ix * cos_r - iy * sin_r + tx;
    double oy = ix * sin_r + iy * cos_r + ty;

    output[2 * i + 0] = ox;
    output[2 * i + 1] = oy;
}