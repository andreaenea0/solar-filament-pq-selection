# public-ensemble weights (team part)

- `lgb_pubown_nometa.txt`: LightGBM selector of public-ensemble, 31 image-only features (no site/year).
  sha256 cbf51d16f6174fe6f2b4d915d36a9a096dae2d0061975643a998f69a6245a61c
- Verifiers: `verifier_resnet18_s{0,1,2}.pt` (timm resnet18.a1_in1k, in_chans=4, 12 epochs on all
  training folds, k14/k15 recipe, retrained on CPU fp32 on 9 Nov because the original state dicts were
  never saved). See the [public-ensemble checksum manifest](../../release_weights_public_ensemble.sha256).
  Test predictions correlate 0.993 with the originals.
- [Parity reference](../../docs/reference/public_ensemble_test.csv) (1,165 filaments), generated with these weights.
- The Git checkout includes the ten Afshar U-Nets and receipt in
  [`weights/public/`](../public). The [replay guide](../../docs/REPRODUCE.md#final-a-replay)
  and [weight licence notice](../../WEIGHTS_LICENSE.md) document their use and attribution.

Map conventions (verified on test, 0 differing pixels vs the submitted maps):
1. Each model: D4-TTA probability -> `np.clip(np.round(p*255), 0, 255).astype(uint8)`.
2. Public: per family, float mean of the 5 fold uint8 maps -> `np.round` -> uint8; public channel =
   `np.round((r34 + effv2s) / 2)` uint8 (numpy round = half to even).
3. Own: `np.round((s0 + s1) / 2)` uint8 over the two k07 seeds.
4. `p = np.stack([pub, own])`; candidates = threshold tree on p (as three-verifiers).
Selection uses 0.35 × LightGBM + 0.65 × mean of the three verifiers,
then `dp_select` with penalty 0.27.
