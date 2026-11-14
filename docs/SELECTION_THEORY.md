# Why the selector uses gain, a penalty and a tree

The deployed selector maximizes an additive **surrogate** on a finite
candidate tree. The dynamic program solves that surrogate exactly.
Learning the gains and choosing the penalty remain statistical problems.

## From record-level PQ to a gain

For image i, let R_i be its number of drawings. Let G be the number of
ground-truth instances pooled across all drawings and images. A selected
mask is compared with each drawing of its image. Predictions must be
pixel-disjoint and a hit requires **IoU > 0.5**, strictly.

First assume each predicted mask hits at most one instance per drawing.
Since predictions are disjoint, a ground-truth instance also cannot have
two strict hits. Define

\[
g_{ic}=\frac{1}{R_i}\sum_{r=1}^{R_i}
  \max_{d\in D_{ir}}\{\operatorname{IoU}(c,d)
       \mathbf{1}[\operatorname{IoU}(c,d)>0.5]\},
\]

where a maximum over an empty drawing is zero. For any feasible selection
S, with \(S_i\) its masks on image i,

\[
A(S)=\sum_i R_i\sum_{c\in S_i}g_{ic},\qquad
K(S)=\sum_i R_i|S_i|,\qquad
\operatorname{PQ}(S)=\frac{2A(S)}{G+K(S)}.
\]

The identity follows from \(G=TP+FN\) and \(K=TP+FP\). These are
**pooled counts**, not a mean of image or drawing PQs.

Adding one disjoint mask to an existing feasible set raises its pooled
PQ q precisely when

\[
\frac{2(A+R_i g)}{G+K+R_i}>\frac{2A}{G+K}
\quad\Longleftrightarrow\quad g>q/2.
\]

This is an add-only statement: it does not decide whether replacing a
parent by several children is beneficial. The tree optimization handles
those conflicts.

## The fractional optimum and its penalty

For a finite feasible family with positive denominator, q* is the best
ratio if and only if

\[
\max_S\{2A(S)-q^*(G+K(S))\}=0.
\]

Indeed every feasible ratio is at most q* exactly when every expression
inside the braces is at most zero, with equality for an optimizer.
At a fixed q, the variable part is

\[
2\sum_iR_i\sum_{c\in S_i}(g_{ic}-q/2).
\]

This is the fractional-programming transformation associated with
[Dinkelbach (1967)](https://doi.org/10.1287/mnsc.13.7.492).
For an oracle with known gains, repeatedly maximizing this expression
and updating q to the resulting ratio solves the finite ratio problem;
stop when the residual is zero (or within a numeric tolerance).
The related calibrated-probability F1 result appears in
[Lipton, Elkan and Narayanaswamy (2014)](https://arxiv.org/abs/1402.1892).
Here g is a quality-weighted strict-hit target, not an ordinary binary
probability, and masks have conflicts.

The release substitutes predicted gains for g and uses a fixed selected
penalty λ. Neither perfect calibration nor the oracle fixed point is
assumed. λ = 0.25 for submitted is an operating point supported by OOF
comparisons. It need not equal its measured PQ divided by two. Maximizing
a ratio of expected additive terms also does not in general maximize the
expectation of a random finite-sample ratio.

An empirical update of lambda to measured PQ(lambda)/2 while selection
still uses imperfect predicted gains is a different procedure. Its
subproblem and update use different objectives, so it has no oracle
Dinkelbach guarantee and can cycle. A fixed number of updates reports a
last-iterate control, not a proved optimum or necessarily a fixed point.

## Why record weights matter

During fitting, a candidate from image i has weight R_i, matching that
image's contribution to the pooled metric. In the oracle expression,
R_i multiplies **every choice on that image**. It therefore cancels when
choosing its best antichain at a fixed λ. It still matters when pooling
PQ, calibrating gains or finding a common λ. Averaging drawing targets
without record-weighted fitting/validation targets a different ratio.

No R_i, annotator, observing date or site is supplied to inference. R_i
and record identifiers are training/evaluation bookkeeping. The deployed
31 features are derived from image pixels and their candidate tree.

## Exact dynamic program for the surrogate

Increasing probability thresholds produce nested connected components.
Two nodes in this forest are either nested or disjoint. A set with no
ancestor/descendant pair (an antichain) is therefore pixel-disjoint.
The proof needs this laminar structure; it does not extend to arbitrary
overlapping masks or a union of unrelated trees.

For candidate c, put \(w_c=\hat g_c-\lambda\). If \(three-verifiers(c)\) is the best
value available in its subtree, then

\[
three-verifiers(c)=\max\left(0,\ w_c,\ \sum_{u\in\mathrm{children}(c)}three-verifiers(u)\right).
\]

Choosing c excludes its descendants. Excluding it leaves disjoint child
subproblems, whose optima add. Induction from the leaves proves the
recurrence; the forest optimum is the sum over roots.
[`dp_select`](../filseg/select.py) evaluates nodes by decreasing level.
On a positive tie it chooses the parent; a zero-valued branch is empty.
The returned set is exact for these supplied scores, not a guarantee of
optimal true PQ. The stored per-subtree index lists may copy a node's
selection through several ancestors; the value recurrence itself uses
one pass over tree edges.

[`tests/test_selection_theory.py`](../tests/test_selection_theory.py)
compares the objective and feasibility with every subset of small random
forests, including negative scores, ties and permuted node indices.

## Overlapping ground-truth drawings: the exception

The original panoptic uniqueness argument assumes non-overlapping
segments in **both** segmentations; see
[Kirillov et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html).
Drawn instances here can overlap. One predicted mask can then hit several
GT masks in one record. The host counts all such hits, rather than solving
a one-to-one assignment. The one-hit identity above is not exact there.

For the general case, define m_rc as the number of strict hits for c in
record r, a_rc as their summed IoU, and b_rc = max(1, m_rc). Disjoint
predictions still prevent a GT mask from hitting two predictions. Thus

\[
\operatorname{PQ}(S)=
\frac{2\sum_{i,r,c\in S_i} a_{rc}}
     {G+\sum_{i,r,c\in S_i} b_{rc}}.
\]

For averaged coefficients \(\bar a_c,\bar b_c\), the oracle parametric
weight becomes \(\bar a_c-(q/2)\bar b_c\). Add-only improvement requires
\(\bar a_c/\bar b_c>q/2\). The same antichain DP remains valid for these
weights. The deployed target takes the **best** strict IoU and assumes
\(\bar b=1\), so it approximates this exceptional case.

The exact host audit of each public-ensemble, three-verifiers and submitted OOF CSV finds one
such record, `010403-20160810234034Lh`: one prediction has two strict hits.
[Observed records](../report/data/multiple_strict_hits.csv).
Evaluation uses [`score_image`](../filseg/metric.py), which preserves both.
The regression test exercises this case explicitly. This note changes
neither the training targets nor the released masks.
