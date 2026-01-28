#!/bin/bash

#SBATCH --output=%j-eureka.out
#SBATCH --job-name=eureka
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dhruvp@cmu.edu

# Navigate to the eureka directory
cd ./Eureka/eureka

# Setup conda environment
source <(curl -sL https://raw.githubusercontent.com/conda/conda/main/conda/shell/etc/profile.d/conda.sh)
conda activate eureka

# Setup poetry environment
## to-do: debug if not working
# source /home/dhruvp/.poetry/env

# Install required packages
# pip install -r requirements.txt

# Set OpenAI API key (make sure to set this in your environment)
# export OPENAI_API_KEY="your-api-key-here"
# export OPENAI_BASE_URL=https://ai-gateway.andrew.cmu.edu

# Run Eureka
# Example: shadow_hand with 4 samples and 2 iterations
cd eureka
python eureka.py env=shadow_hand sample=4 iteration=2 model=gpt-5

# Other example commands:
# python eureka.py env=humanoid sample=16 iteration=5 model=gpt-3.5-turbo-16k-0613
# python eureka.py env=ant sample=16 iteration=5
# python eureka.py env=quadcopter sample=8 iteration=3``