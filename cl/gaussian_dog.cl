__kernel void convoluteX(
	__global uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = mad24(y, src_step, src_offset);
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = OFFSET;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + clamp(x - N / 2 + i, 0, src_cols - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void convoluteY(
	__global uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = src_offset + x;
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = OFFSET;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + src_step * clamp(y - N / 2 + i, 0, src_rows - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void convolute45(
	__global uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = OFFSET;
	for (int i = 0; i < N; i++) {
		int dx = clamp(x - N / 2 + i, 0, src_cols - 1);
		int dy = clamp(y + N / 2 - i, 0, src_rows - 1);
		sum += src[src_offset + src_step * dy + dx] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void convolute135(
	__global uchar *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = OFFSET;
	for (int i = 0; i < N; i++) {
		int dx = clamp(x - N / 2 + i, 0, src_cols - 1);
		int dy = clamp(y - N / 2 + i, 0, src_rows - 1);
		sum += src[src_offset + src_step * dy + dx] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}
