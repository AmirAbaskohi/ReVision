#!/bin/bash

# Extract images from split zip archives
# Processes both Ubuntu and Windows/Mac datasets

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
DATA_DIR="./agentnet_data"

print_header "======================================================================"
print_header "  Step 2: Extracting Images from Split Zip Archives"
print_header "======================================================================"
echo ""

# Check for 7z
if ! command -v 7z &> /dev/null && ! command -v 7za &> /dev/null; then
    print_error "7z not found!"
    echo ""
    print_info "Installing p7zip..."
    
    if command -v brew &> /dev/null; then
        brew install p7zip
    elif command -v apt-get &> /dev/null; then
        apt-get install -y p7zip-full
    else
        print_error "Could not install p7zip automatically."
        print_error "Please install manually:"
        print_error "  macOS: brew install p7zip"
        print_error "  Ubuntu: sudo apt-get install p7zip-full"
        exit 1
    fi
fi

# Determine which 7z command to use
if command -v 7z &> /dev/null; then
    SEVENZ_CMD="7z"
else
    SEVENZ_CMD="7za"
fi

print_info "Using extraction tool: $SEVENZ_CMD"
echo ""

# Function to extract split archives
extract_dataset() {
    local dataset_name=$1
    local zip_dir="${DATA_DIR}/${dataset_name}_images"
    local output_dir="${DATA_DIR}/${dataset_name}_images_extracted"
    
    print_header "Extracting ${dataset_name} images"
    echo ""
    
    if [ ! -d "$zip_dir" ]; then
        print_error "Directory not found: $zip_dir"
        print_error "Run ./1_download_agentnet.sh first!"
        return 1
    fi
    
    if [ ! -f "${zip_dir}/images.zip" ]; then
        print_error "images.zip not found in $zip_dir"
        return 1
    fi
    
    # Count parts
    local total_parts=$(ls "${zip_dir}"/images.z* "${zip_dir}"/images.zip 2>/dev/null | wc -l)
    print_info "📦 Found split archive with ${total_parts} parts"
    print_info "📂 Extracting to: ${output_dir}"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Extract with progress
    print_info "⏳ Extracting... This may take several minutes..."
    echo ""
    
    # Use 7z with progress display
    $SEVENZ_CMD x "${zip_dir}/images.zip" -o"${output_dir}" -y
    
    local exit_code=$?
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        # Count extracted files
        local image_count=$(find "$output_dir" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
        print_info "✅ Extraction complete!"
        print_info "   Extracted ${image_count} image files"
        print_info "   Location: ${output_dir}"
        echo ""
        return 0
    else
        print_error "Extraction failed for ${dataset_name}"
        return 1
    fi
}

# Extract Ubuntu dataset
extract_dataset "ubuntu"

# Extract Windows/Mac dataset
extract_dataset "win_mac"

print_header "======================================================================"
print_header "  ✅ All Extractions Complete!"
print_header "======================================================================"
echo ""
print_info "Extracted datasets:"
print_info "  - ubuntu_images_extracted/"
print_info "  - win_mac_images_extracted/"
echo ""
print_info "Next step: Run python3 3_generate_training_data.py"
echo ""
