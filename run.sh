#!/bin/bash
#SBATCH -A mehakg_llm_psych_0001 
#SBATCH -J mamorx     # job name to display in squeue
#SBATCH -N 1                            # number of nodes
#SBATCH -c 8                # request 8 cpus
#SBATCH -G 1                # request 1 gpu
#SBATCH --time=1:00:00     # 12 hours (up to 48)
#SBATCH --mem=128gb         # request 128GB node memory
#SBATCH --mail-user jerryma@smu.edu     # request to email to your emailID
#SBATCH --mail-type=end                     # request to mail when the model **end**
#SBATCH --error=mamorx_11_4.err          # error print to error_X_%j.log
#SBATCH --output=mamorx_11_4.out        # output print to output_X%j.log

module load gcc cuda cudnn
source ./.venv/bin/activate

echo "Starting DeepSeek-OCR inference and analysis..."
python embedding_visual.py
python latent_representations.py
echo "DeepSeek-OCR inference and analysis completed."