# Performance

- These runs are for `-add=100`.
- Thrashing starts at:

```sh
./hetero-ds-hashtable driver_hashtable_UVM.out -rns=1 -ops=615000000 -add=100 \
      -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin
```

- `./a.out  -ops=620000000 -add=100
  -tra=insert_trace-400e7-100-add-no-dup-SPARSE_UNIQUE.bin   -prefetch all
  -accby-gpu  all`: did not terminate in 10000 s
