FHEdouble rostin(FHEdouble v1, FHEdouble v2) {
  size_t v3 = 10;
  size_t v4 = 0;
  size_t v5 = 1;
  FHEdouble v6 = 2;
  FHEdouble v7 = FHEaddf(v1, v6);
  FHEdouble v8;
  FHEdouble v9 = v7;
  for (size_t v10 = v4; v10 < v3; v10 += v5) {
    FHEdouble v11 = FHEaddf(v9, v2);
    v9 = v11;
  }
  v8 = v9;
  return v8;
}


