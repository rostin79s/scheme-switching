FHEdouble rostin(FHEdouble v1, FHEdouble v2) {
  double v3 = (double)3.00000000000000000e+00;
  double v4 = (double)2.00000000000000000e+00;
  FHEdouble v5 = FHEaddf(v1, v2);
  FHEdouble v6 = FHEmulf(v5, v2);
  FHEdouble v7 = FHEsubf(v6, v1);
  FHEdouble v8 = FHEdivf(v7, v4);
  FHEdouble v9 = FHEaddf(v8, v3);
  return v9;
}


