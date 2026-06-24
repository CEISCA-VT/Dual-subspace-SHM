# Final Results Report

## What are intra vs inter Hamming graphs?
- **Intra-device distances**: Hamming distance between a regenerated ID (at authentication) and its *own* registered ID.
  These show how stable/reproducible each device’s ID is over time.
- **Inter-device distances**: Hamming distances between a regenerated ID and *all other devices’* registered IDs.
  These show how well-separated the devices are (uniqueness).

Ideally: intra distances are low (close to 0), while inter distances are high (close to half the ID length).

## Single-sweep (5 -> 6)
- Devices registered: **300**
- Successfully authenticated: **296**
- Success rate: **98.67%**
- Mean intra Hamming: **7.28**

### Plots
![Single Intra](single_intra.png)

![Single Inter](single_inter.png)

## Multi-sweep (1..6 -> 7)
- Devices registered: **300**
- Successfully authenticated: **300**
- Success rate: **100.00%**
- Mean intra Hamming: **3.72**

### Plots
![Multi Intra](multi_intra.png)

![Multi Inter](multi_inter.png)

