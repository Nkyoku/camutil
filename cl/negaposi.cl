__constant sampler_t samplerLN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void negaposi(
   const image2d_t src,
   __global uchar* dst,
   int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   if (x >= dst_cols) return;
   int dst_index = mad24(y, dst_step, x + dst_offset);
   dst[dst_index] = 255 - read_imageui(src, samplerLN, (int2)(x, y)).r;
};
