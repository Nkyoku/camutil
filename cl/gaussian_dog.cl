__kernel void unsignedConvoluteX(
	__global T *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = mad24(y, src_step, src_offset);
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + clamp(x - N / 2 + i, 0, src_cols - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void unsignedConvoluteY(
	__global T *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = src_offset + x;
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + src_step * clamp(y - N / 2 + i, 0, src_rows - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void signedConvoluteX(
	__global T *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = mad24(y, src_step, src_offset);
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = 128.0f;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + clamp(x - N / 2 + i, 0, src_cols - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}

__kernel void signedConvoluteY(
	__global T *src, int src_step, int src_offset, int src_rows, int src_cols,
	__global uchar *dst, int dst_step, int dst_offset,
	__constant float *coefficients)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int src_index = src_offset + x;
	int dst_index = mad24(y, src_step, dst_offset + x);
	float sum = 128.0f;
	for (int i = 0; i < N; i++) {
		sum += src[src_index + src_step * clamp(y - N / 2 + i, 0, src_rows - 1)] * coefficients[i];
	}
	dst[dst_index] = clamp((int)round(sum), 0, 255);
}
