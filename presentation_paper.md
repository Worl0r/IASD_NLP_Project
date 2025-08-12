The paper attacks a very concrete bottleneck in Transformers: standard self-attention needs O(n2) time and memory with respect to sequence length n, because every token attends to all tokens. This becomes painful as n grows. The authors’ central claim is that the attention matrix is approximately low-rank, so we do not need to compute the full n×n map. Based on this observation, they design Linformer, a modified attention that is linear in n while keeping accuracy close to a regular Transformer.

Key idea : Instead of building attention over all tokens, project keys and values down to a small dimension k≪n, compute attention in this compressed space, and then combine the result—this yields O(nk) cost, which behaves like O(n) if k stays roughly constant.

The paper first argues that the softmax attention matrix P formed inside each head often has a spectrum with a long tail; most mass sits in the top singular values. They show empirical spectra on RoBERTa models where higher layers are even “more low-rank”. Theoretically, using a Johnson–Lindenstrauss style argument, they prove that P can be approximated by a low-rank matrix with rank growing only logarithmically in n (Theorem 1). This justifies trying to replace the big matrix with a factorized form.

We will discuss how Linformer modifies attention. In each head, the model inserts two learned linear projections E and F (size n×k)  before computing attention: project the keys and values from length n down to k, compute an n×k attention map, and multiply by the k-length values. This keeps the number of sequential operations the same as a Transformer, but reduces time and memory per layer from O(n2) to O(nk). A second result (Theorem 2) shows the approximation error can be bounded when k scales with model dimension d (and not with n), which is the reason they can call it “linear”.

For architectural options to reduce cost further. The authors also explore practical tweaks that help when resources are tight:


Projection sharing: share E and F across heads (“headwise”), share key and value projections (“key–value sharing”), or even share a single projection across all layers (“layerwise”). Surprisingly, sharing hurts little in validation perplexity. 

Non-uniform k: use smaller k in higher layers where attention seems lower-rank. 

Alternative projections: replace linear maps with pooling or strided convolutions.

 These are not the core trick, but they make the method simpler and lighter.

They pretrain models with masked-language modeling on BookCorpus + English Wikipedia and then finetune on standard tasks: IMDB and SST-2 (sentiment), QNLI (inference), and QQP (paraphrase). With sequence length n=512, a Linformer using k=128 already tracks Transformer closely in validation perplexity, and with k=256 it even slightly outperforms RoBERTa-base on average across the listed tasks. The trends hold at n=1024 as well. The main message is that performance depends more on k than on n: keeping k fixed while increasing n does not degrade perplexity after convergence.

For Numbers that illustrate the efficiency story. Inference speed and memory footprint improve notably, especially for long inputs. For example, with n=512 and k=128, Linformer is about 1.5× faster and supports 1.7× larger batches than a standard Transformer on a V100 GPU. At longer sequences, gains become dramatic: for n=4096,k=256, they report roughly 3.2× speed-up and 13× memory savings; at n=8192,k=256, speed-up reaches around 5× and memory 26×. These figures are from controlled forward-pass benchmarks with maximum batch sizes that fit into 16 GB.

For the contribution. The paper is not just another sparse attention variant; it reframes self-attention as a low-rank object and then learns the compression as part of the model. Because the projection sits on keys/values rather than on queries, you keep one-step attention (no extra sequential rounds), which is nice for hardware. Also, parameter sharing shows that the added matrices do not explode the parameter count. In short: a simple drop-in change that turns the quadratic term into a linear one while staying competitive in accuracy.

What to keep in mind. The method relies on choosing a reasonable k. Too small k can hurt, too large k make the efficiency lower; the paper’s curves help to pick values like 128 or 256 for moderate sequence lengths. Also, the theory uses distributional assumptions (JL-type arguments) to support low-rank structure; in practice, the spectra plots are the stronger evidence that language modeling attention often behaves this way. Still, for extremely specialized tasks, attention might be less compressible and would deserve a check.

One-sentence takeaway. Linformer shows that if attention is effectively low-rank, then compressing keys/values to a small k gives an O(n) self-attention that keeps Transformer-level quality while cutting both time and memory, especially when sequences are long.
