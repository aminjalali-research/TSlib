# TimeHUT Consolidation Plan

## 📋 **IMMEDIATE ACTIONS** (Execute Today)

### 1. **Remove Obsolete Files**
```bash
cd /home/amin/TSlib/models/timehut
bash cleanup_timehut.sh
```

### 2. **Consolidate Documentation** 
```bash
# Merge useful content from redundant docs
cat README_UNIFIED.md >> README.md  # Merge training examples
# Then remove
rm README_UNIFIED.md

# Archive cleanup docs
mkdir -p archive/old_docs/
mv CLEANUP_*.md INTEGRATION_*.md archive/old_docs/
```

### 3. **Consolidate Training Scripts**
**Priority**: Create unified training interface

**Current State**:
- `train_optimized.py` (1065 lines) - Main script ✅
- `train_optimized_fp32.py` (970 lines) - FP32 variant 
- `train_optimized_fixed_batch.py` (970 lines) - Fixed batch variant
- `train_with_amc.py` (120 lines) - Simple AMC script ✅

**Plan**:
1. **Keep** `train_optimized.py` as main comprehensive script
2. **Keep** `train_with_amc.py` for simple use cases  
3. **Extract** unique features from variants before archiving
4. **Create** `train_unified.py` with all features

---

## 🔍 **FILE ANALYSIS RESULTS**

### ✅ **KEEP** (Essential Files)
```
Core Training:
✅ train_optimized.py          (1065 lines) - Main training script
✅ train_with_amc.py           (120 lines)  - Simple AMC training  
✅ ts2vec.py                   (411 lines)  - Core model
✅ debug_amc_parameters.py     (124 lines)  - Working debug tool
✅ fixed_ablation_study.py     (244 lines)  - Working ablation tool

Core Components:
✅ models/losses_integrated.py (312 lines)  - Unified loss functions
✅ models/encoder.py           (73 lines)   - Model architecture
✅ temperature_schedulers.py   (317 lines)  - Temperature scheduling
✅ datautils.py                (204 lines)  - Data loading
✅ utils.py                    (128 lines)  - Utilities

Documentation:
✅ HUTGuide.md                 (29KB)       - Comprehensive guide
✅ README.md                   (1.6KB)      - Main overview
✅ QUICK_REFERENCE.md          (4KB)        - Quick commands

Analysis Tools:
✅ analysis/core_experiments.py     (690 lines) - Experiment framework
✅ analysis/ablation_studies.py     (902 lines) - Ablation framework
✅ optimization/*.py                          - Optimization tools
```

### 🗂️ **ARCHIVE** (Move to archive/)
```
Old Training Scripts:
📁 train.py                        - Basic version (superseded)
📁 train_timehut_modular.py        - Modular version (functionality merged)

Redundant Documentation:
📁 README_UNIFIED.md               - Merge into README.md first
📁 CLEANUP_*.md                    - Cleanup history docs  
📁 INTEGRATION_*.md                - Integration history docs
📁 PERFORMANCE_OPTIMIZATION_SUMMARY.md - Archive summary
📁 TRAINING_INTEGRATION_COMPLETE.md    - Archive summary

Temporary/Config Files:
📁 ACTION_PLAN.md                  - Planning doc (archive after completion)
📁 INTEGRATED_LOSSES_GUIDE.md      - Merge into main docs
📁 TIMEHUT_CONFIGURATIONS.md       - Merge into HUTGuide.md
```

### ⚠️ **EVALUATE** (Need Decision)
```
Training Variants:
❓ train_optimized_fp32.py         - Check if FP32 features needed
❓ train_optimized_fixed_batch.py  - Check if batch features needed
❓ train_timehut_modular.py        - Check if modular approach needed

Large Scripts:
❓ final_enhanced_baselines.py (1830 lines) - Check if still needed
❓ comprehensive_analysis.py   (676 lines)  - Check vs new analysis tools

Optimization Scripts:
❓ Various optimization/*.py - Keep active ones, archive experiments
```

### ❌ **REMOVE** (Junk/Obsolete)
```
Junk Files:
❌ =1.9.0, =1.10.0              - Version artifacts
❌ 2303.13664v1.pdf             - Research paper (move to docs if needed)

Old Results:
❌ Old ablation_results_*        - Keep only latest results
❌ Old optimization results      - Keep only latest successful runs
```

---

## 🎯 **CONSOLIDATION PRIORITIES**

### **Phase 1: Immediate Cleanup** (Today)
1. Run cleanup script to remove junk and archive old files
2. Consolidate documentation (merge useful content)
3. Identify unique features in training script variants

### **Phase 2: Training Script Consolidation** (This Week)
1. Create comprehensive `train_unified.py` with all features
2. Verify all functionality works
3. Archive old training scripts
4. Update documentation to reference unified script

### **Phase 3: Final Organization** (Next Week)  
1. Optimize directory structure
2. Create final comprehensive documentation
3. Remove remaining redundant files
4. Create reproducible setup instructions

---

## 📊 **EXPECTED OUTCOMES**

**File Count Reduction**:
- **Before**: 13 .md files, 11+ training scripts
- **After**: 3 .md files, 2-3 training scripts

**Space Savings**:
- Archive ~50-70% of files
- Reduce documentation redundancy by ~80%
- Keep only essential, working code

**Improved Usability**:
- Single point of entry for training
- Clear, non-redundant documentation  
- Obvious file purposes and relationships
- Easier maintenance and updates
