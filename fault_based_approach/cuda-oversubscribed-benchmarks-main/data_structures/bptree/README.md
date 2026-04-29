# GPU B-Trees

B-Trees with a key-value semantic.

### Credits

- M. A. Awad, S. Ashkiani, R. Johnson, M. Farach-Colton, and J. D.  Owens,
  “Engineering a high-performance GPU B-Tree".
  [https://doi.org/10.1145/3293883.3295706](https://doi.org/10.1145/3293883.3295706)
- Modified by the SUV (MICRO 2024) authors. (And ported to UVM.)

### TODOs and Porting to UVM

- At some places, `cudaMemcpy` from the earlier implementation is replaced by a
  `memcpy` to the new UVM allocation (`utils.cuh`:`cpyToDevice`).
- TODO: fix the above. Search for `#ifdef SUV_ORIGINAL`.
- TODO: Change keys, values, and the rest to 64 bits. Use signed types to detect
  overflows.
- Values and keys are `[0, 1, 2, 3, 4, ... ]`. Make values random.

- Find out: what happens if keys clash/repeat?
