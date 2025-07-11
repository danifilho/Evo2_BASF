#!/usr/bin/env bash
# generate_windows_chr4.sh  <window_size_bp>
# Creates: windows_<SIZE>/NC_003075.7_<start>_<end>.fa  (one file per window)

set -euo pipefail

# 1) Configuring the paths
CHR4_FASTA="/mnt/gs21/scratch/dasilvaf/evo2_arabidopsis/GCF_000001735.4_TAIR10.1_genomic.fna"   # I did created this testing with just the Chr4
BEDTOOLS=$(command -v bedtools)


# 2) Arg parsing
[[ $# -ne 1 ]] && { echo "Usage: $0 <window_size_bp>"; exit 1; }
WIN=$1
OUTDIR="windows_${WIN}_full"
BEDFILE="${OUTDIR}/chr4_${WIN}.bed"

mkdir -p "$OUTDIR"

# 3) Building BED of contiguous windows

$BEDTOOLS makewindows -g "$CHR4_FASTA.fai" -w "$WIN" -s "$WIN" \
  | awk -v OFS="\t" '{print $1,$2,$3, $1"_"$2"_"$3}' > "$BEDFILE"

echo "Creating BED for ${WIN}-bp windows …"

# 4) Extracting multi-FASTA, then splitting into individual files

echo "Extracting FASTA …"
MULTIFASTA="${OUTDIR}/chr4_${WIN}.fa"
$BEDTOOLS getfasta -fi "$CHR4_FASTA" -bed "$BEDFILE" -name -fo "$MULTIFASTA"

echo "Splitting multi-FASTA into individual files …"
awk -v outdir="$OUTDIR" '
    /^>/ {
        split($0, a, "::")                # a[1] = ">NC_003075.7_0_8192"
        fname = substr(a[1], 2) ".fa"     # drop leading ">"
        f = outdir "/" fname
        next
    }
    { print > f }
' "$MULTIFASTA"

rm "$MULTIFASTA"
echo "DONE – FASTA windows in $OUTDIR/"
