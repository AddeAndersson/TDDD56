/*
 * Placeholder OpenCL kernel
 */


static void exchange(__global unsigned int *i, __global unsigned int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__kernel void bitonic(__global unsigned int *data,int j, int k)
{
  int id = get_global_id(0);
  int ixj;

  ixj = id^j; // Calculate indexing!

  if ((ixj)>id)
  {
    if ((id&k)==0 && data[id]>data[ixj]) exchange(&data[id],&data[ixj]);
    if ((id&k)!=0 && data[id]<data[ixj]) exchange(&data[id],&data[ixj]);
  }
}