#!/bin/bash

# Complete pipeline to prepare AgentNet data for Qwen training
# Processes ALL samples from BOTH datasets (Ubuntu + Windows/Mac)

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo ""
print_header "======================================================================"
print_header "  AgentNet to Qwen Training Data Pipeline"
print_header "======================================================================"
echo ""
print_info "This pipeline will process ALL samples from BOTH datasets:"
print_info "  - Ubuntu dataset: ~5K samples (~73GB images)"
print_info "  - Windows/Mac dataset: ~18K samples (~108GB images)"
print_info "  - Total: ~23K training samples (~181GB)"
echo ""
print_warning "Make sure you have sufficient disk space!"
echo ""
print_header "======================================================================"
echo ""

# Check if user wants to skip steps or set window size
SKIP_DOWNLOAD=false
SKIP_EXTRACT=false
WINDOW_SIZE=3

for arg in "$@"; do
    if [ "$arg" = "--skip-download" ]; then
        SKIP_DOWNLOAD=true
    elif [ "$arg" = "--skip-extract" ]; then
        SKIP_EXTRACT=true
    elif [[ "$arg" =~ ^--window-size=(.+)$ ]]; then
        WINDOW_SIZE="${BASH_REMATCH[1]}"
    fi
done

# Install Python dependencies
print_info "Installing Python dependencies..."
pip3 install -q tqdm Pillow
echo ""

# Step 1: Download
if [ "$SKIP_DOWNLOAD" != true ]; then
    print_header "STEP 1/4: Downloading AgentNet Data"
    echo ""
    ./1_download_agentnet.sh
else
    print_info "⏭️  Skipping download step"
    echo ""
fi

# Step 2: Extract
if [ "$SKIP_EXTRACT" != true ]; then
    print_header "STEP 2/4: Extracting Images"
    echo ""
    ./2_extract_images.sh
else
    print_info "⏭️  Skipping extraction step"
    echo ""
fi

# Step 3: Generate training data
print_header "STEP 3/4: Generating Training JSON Files"
echo ""
python3 3_generate_training_data.py

# Step 4: Windowing
print_header "STEP 4/4: Windowing Samples (Window Size: $WINDOW_SIZE)"
echo ""
python3 4_windowing_samples.py --window-size="$WINDOW_SIZE"

# Final summary
echo ""
print_header "======================================================================"
print_header "  🎉 Pipeline Complete!"
print_header "======================================================================"
echo ""
print_info "Your training data is ready!"
echo ""
print_info "📁 Training data locations:"
print_info "   - agentnet_data/ubuntu_training_data/     (~5K samples)"
print_info "   - agentnet_data/win_mac_training_data/   (~18K samples)"
print_info "   - agentnet_data/windowed_training_data/  (windowed samples)"
echo ""
print_info "📊 Total: ~23K training samples in Qwen format"
echo ""
print_info "Windowing configuration:"
print_info "   - Window size: $WINDOW_SIZE (most recent screenshots)"
echo ""
print_info "Each JSON file contains:"
print_info "   - messages: Multi-turn conversation format"
print_info "   - images: Paths to screenshot images"
print_info "   - metadata: Task completion info and scores"
echo ""
