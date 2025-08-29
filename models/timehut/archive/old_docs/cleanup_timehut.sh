#!/bin/bash

# TimeHUT Folder Cleanup Script
# Run from /home/amin/TSlib/models/timehut

echo "🧹 TimeHUT Cleanup Starting..."

# Create archive directory
mkdir -p archive/
mkdir -p archive/old_training_scripts/
mkdir -p archive/old_docs/
mkdir -p archive/old_results/

# 1. Remove junk files
echo "Removing junk files..."
rm -f =1.9.0 =1.10.0
# Keep PDF in docs if it's relevant research
# mv 2303.13664v1.pdf archive/ 

# 2. Archive old training scripts
echo "Archiving redundant training scripts..."
mv train.py archive/old_training_scripts/ 2>/dev/null || echo "train.py not found"
mv train_timehut_modular.py archive/old_training_scripts/

# Keep these for now, will consolidate later:
# mv train_optimized_fp32.py archive/old_training_scripts/
# mv train_optimized_fixed_batch.py archive/old_training_scripts/

# 3. Archive redundant documentation
echo "Archiving redundant documentation..."
mv CLEANUP_*.md archive/old_docs/
mv INTEGRATION_*.md archive/old_docs/
mv PERFORMANCE_OPTIMIZATION_SUMMARY.md archive/old_docs/
mv TRAINING_INTEGRATION_COMPLETE.md archive/old_docs/

# 4. Clean old result directories (keep latest)
echo "Cleaning old result directories..."
find . -name "*results_202508*" -type d | head -n -3 | xargs -I {} mv {} archive/old_results/

# 5. Clean __pycache__ directories
echo "Cleaning Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 6. Remove empty directories
echo "Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📊 Space saved:"
du -sh archive/
echo ""
echo "🗂️  Remaining structure:"
ls -la *.py *.md | wc -l
echo " Python files and documentation remaining"
