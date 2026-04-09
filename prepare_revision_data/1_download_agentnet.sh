#!/bin/bash

# Download AgentNet data from HuggingFace
# Downloads both Ubuntu and Windows/Mac datasets

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

# Configuration
DATA_DIR="./agentnet_data"
REPO_ID="xlangai/AgentNet"

mkdir -p "$DATA_DIR"

print_header "======================================================================"
print_header "  Step 1: Downloading AgentNet Data from HuggingFace"
print_header "======================================================================"
echo ""
print_info "Repository: $REPO_ID"
print_info "Output directory: $DATA_DIR"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    print_warning "huggingface-cli not found. Installing..."
    pip3 install -q huggingface_hub
fi

# Download Ubuntu dataset
print_header "Downloading Ubuntu Dataset (5K samples)"
echo ""
print_info "📥 Downloading agentnet_ubuntu_5k.jsonl..."
huggingface-cli download "$REPO_ID" \
    agentnet_ubuntu_5k.jsonl \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

print_info "📥 Downloading ubuntu_images/ (split zip files, ~73GB)..."
print_warning "This will take a while depending on your connection..."
huggingface-cli download "$REPO_ID" \
    --include "ubuntu_images/*" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

echo ""
print_info "✅ Ubuntu dataset downloaded"
echo ""

# Download Windows/Mac dataset
print_header "Downloading Windows/Mac Dataset (18K samples)"
echo ""
print_info "📥 Downloading agentnet_win_mac_18k.jsonl..."
huggingface-cli download "$REPO_ID" \
    agentnet_win_mac_18k.jsonl \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

print_info "📥 Downloading win_mac_images/ (split zip files, ~108GB)..."
print_warning "This will take a while depending on your connection..."
huggingface-cli download "$REPO_ID" \
    --include "win_mac_images/*" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

echo ""
print_info "✅ Windows/Mac dataset downloaded"
echo ""

print_header "======================================================================"
print_header "  ✅ Download Complete!"
print_header "======================================================================"
echo ""
print_info "Downloaded files:"
print_info "  - agentnet_ubuntu_5k.jsonl"
print_info "  - ubuntu_images/ (split archives)"
print_info "  - agentnet_win_mac_18k.jsonl"
print_info "  - win_mac_images/ (split archives)"
echo ""
print_info "Next step: Run ./2_extract_images.sh to extract the images"
echo ""
