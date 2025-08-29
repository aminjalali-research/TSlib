#!/bin/bash

# Final TimeHUT Consolidation
# Moves training script variants to archive after confirming they're just benchmarking versions

cd /home/amin/TSlib/models/timehut

echo "🔧 Final TimeHUT Consolidation"
echo "================================"

# Archive training script variants (they're just benchmarking versions)
echo "Archiving training script variants..."
mv train_optimized_fp32.py archive/old_training_scripts/ 2>/dev/null || echo "fp32 already archived"
mv train_optimized_fixed_batch.py archive/old_training_scripts/ 2>/dev/null || echo "fixed_batch already archived"

# Clean up large analysis files if they're duplicates
echo "Checking large analysis files..."
if [ -f "final_enhanced_baselines.py" ]; then
    echo "⚠️  Large file found: final_enhanced_baselines.py (1830 lines)"
    echo "   Consider archiving if superseded by analysis/ framework"
    # mv final_enhanced_baselines.py archive/old_training_scripts/
fi

# Final count
echo ""
echo "📊 Final file structure:"
echo "Python scripts:"
find . -name "*.py" -not -path "./archive/*" | wc -l
echo "Documentation:"
find . -name "*.md" -not -path "./archive/*" | wc -l

echo ""
echo "✅ Consolidation complete!"
echo ""
echo "🎯 Main entry points:"
echo "  📈 Training: train_optimized.py (comprehensive) or train_with_amc.py (simple)"
echo "  🧪 Testing: debug_amc_parameters.py, fixed_ablation_study.py"  
echo "  📚 Docs: HUTGuide.md (comprehensive), README.md (overview)"
echo ""
echo "📁 Archived files:"
du -sh archive/
