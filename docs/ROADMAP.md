# FDT v4.1 Roadmap

1. Preserve Stage A, the selected Exact Pointer checkpoint, and both rejected
   loop-training interventions as immutable evidence.
2. Keep the repaired generator contract fixed: explicit-only Exact copy,
   generated-only loop control, and full recompute at intermediate RoPE blend.
3. Design one fresh isolated raw-generation objective pilot. It must target
   actual model-generated failure trajectories and pass penalty-off generation,
   V20 paired capability, validation, routing, cache, and Exact Memory gates.
4. Do not resume the rejected unlikelihood checkpoint or learned controller.
5. Advance staged RoPE and the token-normalized 512/2K/4K/8K/16K curriculum
   only after the isolated raw-generation gate passes.
6. Keep natural language and factual knowledge primary, retrieval substantial,
   and code/JSON auxiliary. Official evaluation remains unquantized FP32.
7. Treat conversational training and fuzzy-space reasoning as later stages after
   the base model is materially usable.
