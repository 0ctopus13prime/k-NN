# Motivation
With Scalar Quantized (i.e. using faiss engine with 32x compression), Approximated <q, x> (i.e. q is query vector and x is data vector) can be achieved with
<q', Q(x')> + <q, c> + <c, x> - <c, c> where q' = q - centroid, x' = x - centroid, Q(x') is quantized vector of x'.
This approximation is being made in the 1st phase during HNSW graph searching.

Now, we can introduce residual error vector 'r' as follow:
x' = Q(x') + r

Then the true value of <q, x> is defined as <q', Q(x')> + <q, c> + <c, x> - <c, c> + <q', r>.

# Idea : Use Quantized `r` for results refinement.
We can first start with 4 bit quantization of `r` for <q', r>.

## 2 phase search and rescoring
We are adopting 2-phase search with oversampling factor (we're using 2) where performing 1st phase search with quantized vectors first,
then load full precision vectors to refine and narrow down candidates to top-k.

However, if we could find a way to use <q', r> for refinement instead of doing <q, x> for better performance, then I'm seeing two benefits.
1. Lesser page faults. In memory constrained environment, it will require fewer read disk IO as r is 4 bit quantized, thus compared to full precision vectors loading, IO should be less.
For example, dimension 768 would require 3072 bytes per a single vector, but with 4bits error residual quantization, it would need 384 bytes for IO.
Technically, 4kb is the unit for IO, so it will need 4kb IO anyway, but max possible vectors per IO will increase for sure.
2. When all present in memory, I'm expecting <q', r> would be faster than <q, x> with SIMD.

# Goal
- We first validate this idea, especially listed two benefits above.
- We should find a way to how we can save error residual vectors separately.
- We should find a way to use error residual vectors for result refinement.
