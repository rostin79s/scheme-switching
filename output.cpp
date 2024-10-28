FHEdouble rostin(FHEdouble v1, FHEdouble v2) {
  size_t v3 = 10;
  size_t v4 = 0;
  size_t v5 = 1;
  FHEdouble v6 = 1;
  FHEdouble v7 = 0;
  FHEdouble v8;
  FHEdouble v9;
  FHEdouble v10 = v2;
  FHEdouble v11 = v1;
  for (size_t v12 = v4; v12 < v3; v12 += v5) {
    FHEdouble v13 = FHEaddf(v11, v10);
    FHEdouble v14 = FHEoeqf(v13, v10);
    FHEdouble v15 = FHEselectf(v14, v6, v7);
    FHEdouble v16 = FHEaddf(v10, v15);
    v10 = v16;
    v11 = v13;
  }
  v8 = v10;
  v9 = v11;
  return v8;
}


