# trueSV
**trueSV: Trustworthy Deep Learning-based STructural Variant Calling**

trueSV is a deep learning-based tool for detecting structural variants, excelling at low coverage (5×-10×). 



## 🧭 Features
- Detects insertions, deletions, inversions, duplications, and inverted duplications using YOLOv11m deep learning
- Maintains high accuracy even at low sequencing coverage  
- High accuracy for large SVs
- This version supports PacBio HiFi, PacBio CLR, and ONT data 
- Supports GPU acceleration and multiprocessing


## 🚀 Installation
```bash
pip install "git+https://github.com/sajadtavakoli/trueSV.git"

```

## 🧩 Usage
### Quick usage
```bash
trueSV -a aln.bam -r ref.fa -o output.vcf -c 10
```

c is the coverage of the sample

### All options
```bash
# For example, we want to call SVs in Chromosomes 1 and 2 (chr1 and chr2) of sample NA19240, which has 10X coverage. 
trueSV \
    -a path/to/aln.bam \ 
    -r path/to/reference.fa \ 
    -o path/to/output.vcf \ 
    -c 10 \ 
    -s NA19240 \ 
    -t 8 \ 
    --contigs chr1,chr2 \  
``` 

| Flag                  | Name                | Type  | Default  | Description                                                                                    |
| --------------------- | ------------------- | ----- | -------- | ---------------------------------------------------------------------------------------------- |
| `-a`, `--aln_path`    | Alignment file path | `str` | —        | Path to input alignment file (`.bam` or `.cram`)                                               |
| `-r`, `--ref_path`    | Reference file path | `str` | —        | Path to reference genome file (`.fa`)                                                          |
| `-o`, `--out_path`    | Output file path    | `str` | —        | Path to output VCF file (`.vcf`)                                                               |
| `-c`, `--coverage`    | Sample coverage     | `int` | —        | Sequencing coverage (e.g., `10` for 10×)                                                       |
| `-s`, `--sample`      | Sample name         | `str` | `SAMPLE` | Name of the sample used for labeling output                                                    |
| `-t`, `--threads`     | Threads             | `int` | `4`      | Number of CPU threads to use                                                                   |
| `--contigs`           | Contigs of interest | `str` | `all`    | Comma-separated list of contigs (e.g., `chr1,chr2`)                                            |

