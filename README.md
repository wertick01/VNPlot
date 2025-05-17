## VNPlot - a small software add-in for processing and visualizing VirusNeutralization data

We hope that this code will help simplify data processing and make your graphs for articles a little better :).
Basic calculations are performed using the [Neutcurve](https://jbloomlab.github.io/neutcurve/) [1] library, after which an add-in is applied to manipulate graph styles.

The code/program accepts a file or files from a folder (csv, csv, xls, xlsx tabular format) as input in the following format:
| 7-35-L15H | 57-35-L15H | 58-35-L15H | 57-18-7-Fc | 58-18-7-Fc |  SA55  |     X     |
|-----------|------------|------------|------------|------------|--------|-----------|
| 5.2342    | 6.6291     | 3.4411     | 5.0134     | 8.9558     | 4.9267 | 8000      |
| 5.6616    | 6.5393     | 5.3159     | 4.1570     | 8.1871     | 4.4405 | 2666.6667 |
| 5.0297    | 7.6328     | 8.6859     | 4.3137     | 12.0099    | 6.2767 | 888.8889  |
| 5.0655    | 21.2298    | 22.5824    | 4.2992     | 21.2612    | 7.8086 | 296.2963  |
| 7.0585    | 20.9435    | 19.4041    | 4.4615     | 20.8780    | 10.9575| 98.7654   |
| 17.9033   | 32.0157    | 27.6074    | 12.6721    | 28.5181    | 14.847 | 32.9218   |
| 26.8616   | 32.2663    | 32.4118    | 26.5669    | 30.7874    | 19.8624| 10.9739   |
| 26.3977   | 35.4702    | 34.6453    | 34.1568    | 34.7000    | 25.1027| 3.6580    |
| 27.5391   | 33.9185    | 34.3822    | 35.3457    | 36.3479    | 30.2611| 1.2193    |
| 31.0484   | 37.5901    | 35.2967    | 38.5979    | 38.5581    | 33.1895| 0.4064    |

where X is the concentration of the antibody in each titration.

For the example watch file all_release_version.ipynb

1. Loes AN, Tarabi RAL, Huddleston J, Touyon L, Wong SS, Cheng SMS, Leung NHL, Hannon WW, Bedford T, Cobey S, Cowling BJ, Bloom JD.2024.High-throughput sequencing-based neutralization assay reveals how repeated vaccinations impact titers to recent human H1N1 influenza strains. J Virol98:e00689-24.https://doi.org/10.1128/jvi.00689-24