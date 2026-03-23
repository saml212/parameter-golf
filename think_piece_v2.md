## Takeaways So Far: Constraints and Collaboration

### Abstract

This contest changed my mind about what a research competition can do. The constraints do more than make the game fair: they push effort away from ordinary training tweaks and toward evaluation, quantization, compression, and other underexplored parts of the pipeline. Public PRs, synthesis agents, and maybe shared weights then turn isolated runs into a collective search process.

Four days ago, 1.15 BPB looked almost impossible because the 4-hour unlimited-compute run was only 1.175. I'm at 1.1320 now, and the surprise is not just how fast the numbers moved but that this turned out to be less a training contest than a search across the whole pipeline.

### The Hunt for Leverage

I started where everyone starts: learning rates, gradient clipping, warmdown schedules. That led to one real finding. The post-training int8 quantization penalty was bigger than all my hyperparameter gains combined. The thing destroying my score wasn't the model. It was the compression step everyone was treating as an afterthought. That became [PR #61](https://github.com/openai/parameter-golf/pull/61).

But mattqlf's [PR #50](https://github.com/openai/parameter-golf/pull/50), which landed before mine, made the deeper point. The stock baseline got to 1.1925 with zero training changes, just by changing evaluation. That was the moment the contest stopped looking like a training contest and started looking like a search across the whole pipeline.

My own path more or less followed that lesson: [PR #96](https://github.com/openai/parameter-golf/pull/96) pushed evaluation and long-context training, [PR #114](https://github.com/openai/parameter-golf/pull/114) came from revisiting quantization with a different setup, [PR #236](https://github.com/openai/parameter-golf/pull/236) moved into batch-size and architecture tradeoffs, and [PR #332](https://github.com/openai/parameter-golf/pull/332) bought more depth through compression.

### Where the Leverage Lives

One rule survived every phase of ~130 experiments: step throughput is king. On a 10-minute budget, a technique that slows each step too much usually loses, even if it is better in isolation.

The second rule is that negative results matter, but only locally. Shared mistakes create norms and save compute, but those norms are conditional statements about a regime, not laws of nature.

That was the int6 story for me. Early on I tried int6 quantization in a bad setup, got a terrible result, and wrote it off. Later I came back with per-row scaling and a different LR regime, and it became the foundation of [PR #114](https://github.com/openai/parameter-golf/pull/114). The same thing happened with eval@2048: huge win on 1xH100, not a win on 8xH100. The lesson wasn't to ignore the community. It was to learn the rule well enough to know when it no longer applies.

The leverage migrates. When everyone optimizes training, it moves to evaluation. When everyone has sliding window, it moves to compression. When everyone has int6, it moves to architecture. jfprincz's [PR #287](https://github.com/openai/parameter-golf/pull/287) and [PR #315](https://github.com/openai/parameter-golf/pull/315) are good examples.

### What Constraints Are For

The non-obvious thing about contests like this is that the constraints are not just there to make the game fair. They decide which forms of ingenuity become visible. If you want progress in evaluation, compression, or efficiency, one way to get it is to make training compute boring and non-negotiable, then watch where the leverage moves. More broadly, if a field already knows how to scale one part of the pipeline, you can learn a lot by freezing that part and forcing attention elsewhere. Parameter Golf is not just a contest. It is a way of steering search.

### Collaboration as Infrastructure

The runs themselves are not the main bottleneck. The real bottleneck is the gap between runs: deciding what to try next, figuring out which negative results to trust, and turning the community's partial findings into something usable. Public PRs help. [Issue #140](https://github.com/openai/parameter-golf/issues/140) turned hundreds of PRs into a living research document. And a shared Hugging Face weights repo, as proposed in [Issue #202](https://github.com/openai/parameter-golf/issues/202), would tighten that loop even further by turning each submission from an expensive recipe to rerun into an artifact other people could immediately inspect, compare, and build on.

The most valuable artifact from this contest might not be any single model. It might be the coordination stack: public PRs, synthesis agents, searchable negative results, and reusable weights that let a community search as if it were one lab.

Someone trained on the validation set and got 1.01. The organizers patched the rules. But that number still says something important: the remaining gap is not obviously just about raw model capacity.

### Appendix: Where I'd Look Next

- **Evaluation:** KV-cache reuse, smaller effective strides, neural-plus-classical compression mixtures.
- **Quantization and compression:** pre-quant rotations, simple post-quant calibration, better entropy coding.
- **Cheap architecture:** value residuals, memory tokens, backout-style connections.
- **Collaboration itself:** better synthesis, better PR search, shared weights, easier artifact reuse.

I'm not done. There are still blank rows in the leverage table.

---

*~130 experiments. ~45 H100-hours. 5 PRs ([#61](https://github.com/openai/parameter-golf/pull/61), [#96](https://github.com/openai/parameter-golf/pull/96), [#114](https://github.com/openai/parameter-golf/pull/114), [#236](https://github.com/openai/parameter-golf/pull/236), [#332](https://github.com/openai/parameter-golf/pull/332)). Some documentation prepared with AI assistance. All mistakes are my own.*
