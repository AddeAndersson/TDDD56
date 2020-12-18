/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  int gid = get_global_id(0);
  int lid = get_local_id(0);

  for(int i = ceil(get_local_size(0)/2.0); i >= 1; i = i/2) {
      if(lid < i) {
          data[gid] = max(data[gid + i], data[gid]);
      }
  }

  if(lid == 0) {
      data[get_group_id(0)] = data[gid];
  }
}