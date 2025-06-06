  GNU nano 6.2                                                                                         h200_slurm.sh                                                                                                  
#!/bin/bash --login
#SBATCH --job-name=evo2_20000_2_full          # change if you benchmark other sizes
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1            # 1 × NVIDIA H200
#SBATCH --cpus-per-gpu=40                 # plenty for I/O + tokeniser
#SBATCH --mem=64G                         # adjust if you need more RAM
#SBATCH --time=4:00:00                    # queue limit on “short”
#SBATCH -o "/mnt/gs21/scratch/dasilvaf/evo2/logs/stdout.%x.%j.%N"
#SBATCH -e "/mnt/gs21/scratch/dasilvaf/evo2/logs/stderr.%x.%j.%N"

###############################################################################
# USER-ADJUSTABLE PATHS
###############################################################################
WORKDIR=$PWD                               # directory where you launch sbatch
SIF=$WORKDIR/evo2_latest.sif               # container image
SCRIPT=$WORKDIR/embed_evo2.py              # updated wrapper
HFCACHE=$WORKDIR/huggingface               # holds HF auth token / model cache
INDIR=$WORKDIR/windows_20000_full           # input FASTA windows (bind mount)
OUTDIR=$WORKDIR/embeddings_20000_full       # will receive .pt tensors

echo "============================================================"
echo "JOB  START : $(date --iso-8601=seconds)"
echo "NODE       : $(hostname)"
echo "SLURM ID   : $SLURM_JOB_ID"
echo "============================================================"

###############################################################################
# MAIN WORK
###############################################################################
singularity exec --nv --writable-tmpfs --pwd /workspace \
  -B "$HFCACHE:/root/.cache/huggingface" \
  -B "$INDIR:/workspace/windows_20000_full" \
  -B "$OUTDIR:/workspace/embeddings_20000_full" \
  -B "$SCRIPT:/workspace/embed_evo2.py" \
  "$SIF" \
  python3 /workspace/embed_evo2.py \
        --indir  /workspace/windows_20000_full \
        --outdir /workspace/embeddings_20000_full
STATUS=$?

###############################################################################
# RESOURCE-USAGE SUMMARY
###############################################################################
echo "============================================================"
echo "RESOURCE SUMMARY (sacct)"
sacct -j "$SLURM_JOB_ID" \
      --units=M \
      --format=JobIDRaw,Elapsed,MaxRSS,MaxVMSize,AveRSS,AveCPU,MaxRSSNode,AllocTRES,ReqTRES%30 \
      --parsable2 2>/dev/null \
  || {
        echo "sacct data not yet available – falling back to sstat"
        sstat -j "${SLURM_JOB_ID}.batch" \
              --format=MaxRSS,MaxVMSize,AveRSS,AveCPU,MaxDiskRead,MaxDiskWrite \
              --units=M
     }

echo "EXIT CODE  : $STATUS"
echo "JOB  END   : $(date --iso-8601=seconds)"
echo "============================================================"
