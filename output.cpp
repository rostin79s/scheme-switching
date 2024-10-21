FHEdouble rostin(FHEdouble v1, FHEdouble v2) {
  double v3 = (double)2.00000000000000000e+00;
  FHEdouble v4 = FHEaddf(v1, v2);
  FHEdouble v5 = FHEmulf(v4, v2);
  FHEdouble v6 = FHEsubf(v5, v1);
  FHEdouble v7 = FHEdivf(v6, v3);
  return v7;
}


