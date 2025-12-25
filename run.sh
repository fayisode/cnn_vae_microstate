#!/bin/bash
#SBATCH --partition=datamite
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu:1|gpu:2"  # ADD THIS LINE - restricts to GPU1 and GPU2
#SBATCH --job-name=my_training
#SBATCH --time=24:00:00
#SBATCH --mem=100G

# if [ -f "outputs" ]; then
# 	echo "Removing existing outputs..."
# 	rm -r outputs
# fi

# Ensure Poetry is installed, skip if already installed
if ! command -v poetry &>/dev/null; then
	echo "Poetry not found. Installing..."
	curl -sSL https://install.python-poetry.org | python3 -

	# Add Poetry to PATH for this session
	export PATH="$HOME/.poetry/bin:$PATH"

	# Check if Poetry is now available
	if ! command -v poetry &>/dev/null; then
		echo "Poetry installation may have completed, but the command is not in PATH."
		echo "Trying alternative location..."
		export PATH="$HOME/.local/bin:$PATH"
	fi
else
	echo "Poetry is already installed."
fi

# Verify Poetry installation
if ! command -v poetry &>/dev/null; then
	echo "ERROR: Poetry command still not found after installation."
	echo "Please install Poetry manually and try again."
	exit 1
fi

# Remove poetry.lock if it exists
if [ -f "poetry.lock" ]; then
	echo "Removing existing poetry.lock..."
	rm poetry.lock
fi

if [ -f "s3vdx_test4_3.pth" ]; then
	echo "Removing existing model..."
	rm s3vdx_test4_3.pth
fi

# if [ -f "outputs" ]; then
# 	echo "Removing existing outputs..."
# 	rm -r outputs
# fi

if [ -r "images" ]; then
	echo "Removing existing images..."
	rm -r images
fi

if [ -r "visualizations" ]; then
	echo "Removing existing visualizations..."
	rm -r visualizations
fi

# if [ -r "outputs" ]; then
# 	echo "Removing existing outputs..."
# 	rm -r outputs
# fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Define the file and arguments separately
PYTHON_FILE="./train.py"
RUN_ID="eeg_hyperparam_search_v1_s01"
PARTICIPANT="s01"

# Execute the command
# Execute the specified Python file
echo "Running $PYTHON_FILE..."
poetry run python "$PYTHON_FILE" --participant "$PARTICIPANT" --run_id "$RUN_ID"
