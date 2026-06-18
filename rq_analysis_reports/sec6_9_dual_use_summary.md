
## Section 6: Identity Analysis
- Separation ratio (inter/intra): 34.83x
- EER: 0.0087
- Best identification accuracy: 98.4% with 50 PCs
- Top-5 identity PCs (ANOVA): [2, 3, 0, 9, 10]
- TAR @ 1% FAR: 0.636

## Section 7: Health Monitoring
- 3-way condition classification (temp/aging/load): 0.643
- Baseline vs condition detection: 0.597
- Per-type baseline detection — temp: ['0.750'], aging: ['0.710'], load: ['0.740']
- Top-5 health PCs (ANOVA): [97, 63, 74, 77, 61]

## Section 8: Subspace Separation
- Identity subspace: 27 PCs (sorted by I/H ratio gap)
- Health subspace: 73 PCs
- Dual-use PCs (both > median): 26 — [np.int64(5), np.int64(7), np.int64(11), np.int64(14), np.int64(18), np.int64(19), np.int64(20), np.int64(21)]
- Gap threshold: 0.097

## Section 9: Dual-Use Framework
- Auth EER: 0.2229 (identity subspace)
- Health detection EER: 0.2089 (health subspace)
- Health deviation by condition:
  - aging: 8.909 ± 5.781 (n=1180)
  - baseline: 4.373 ± 0.929 (n=1770)
  - loading: 74.864 ± 60.597 (n=1180)
  - temperature: 18.866 ± 8.273 (n=885)
