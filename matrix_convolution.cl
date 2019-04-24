__kernel void matrix_convolution(__global double *a,
                         __global double *b,
                         __global double *c,
                         int n, int m) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  int hm = (m - 1) / 2;
  if (row >= n || column >= n) {
    return;
  }
  double sum = 0;
  for (int i = -hm; i <= hm; i++) {
    int current_row = i + row;
    if (current_row < 0 || current_row >= n) 
	  continue;
	for (int j = -hm; j <= hm; j++) {
	  int current_column = j + column;
	  if (current_column < 0 || current_column >= n) 
	     continue;
	  sum += a[current_row * n + current_column] * b[(i + hm) * m + j + hm];
	}
  }
  c[row * n + column] = sum;
}
