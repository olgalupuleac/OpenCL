#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void prefix_sum(__global double* input, __global double* output, __local double* a, __local double* b, int n)
{
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint block_size = get_local_size(0);


  if (gid < n)
  {
    a[lid] = b[lid] = input[gid];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint s = 1; s < block_size; s <<= 1)
    {
      if(lid > (s - 1))
      {
        b[lid] = a[lid] + a[lid - s];
      }
      else
      {
        b[lid] = a[lid];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      SWAP(a, b);
    }
    if (gid < n)
    {
      output[gid] = a[lid];
    }
}


__kernel void final_sum(__global double* input,  __global double* sums, __global double* output, int n)
{
  uint gid = get_global_id(0);
  uint block_size = get_local_size(0);

  if (gid < n)
  {
    output[gid] = input[gid] + sums[gid / block_size];
  }
}