# Layer Implementation TODOs

## Future Enhancements

### Kernel Count Sophistication (base.py line 146)
Make kernel count more sophisticated to account for:
- Different fusion strategies (e.g., flash attention vs naive)
- Hardware-specific fusion (some GPUs fuse better than others)
- Batch size effects (some kernels only launch once regardless of batch)
- Dynamic fusion decisions based on input shapes
Currently using conservative static estimates per layer type.

### Complex Memory Hierarchies (base.py line 371)
Add support for complex memory hierarchies (e.g., TPUs):
- Specify where weights live vs where KV cache lives
- Multi-tier memory (SRAM, HBM, DRAM)
- Memory hierarchy-aware bandwidth calculations
Currently defaulting to HBM for all memory operations.