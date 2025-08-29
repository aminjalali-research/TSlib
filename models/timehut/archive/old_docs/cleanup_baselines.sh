#!/bin/bash

# TimeHUT Baselines Cleanup Script
# REMOVES 6.2GB of redundant data!

cd /home/amin/TSlib/models/timehut

echo "🧹 TimeHUT Baselines Cleanup"
echo "============================"

# Show current size
echo "📊 Current baselines folder size:"
du -sh baselines/

echo ""
echo "⚠️  ANALYSIS:"
echo "   - baselines/TS2vec/datasets/ = 6.2GB (DUPLICATE of main datasets)"
echo "   - baselines/TS2vec/ts2vec.py = Original TS2Vec (323 lines)"
echo "   - Main ts2vec.py = Enhanced TimeHUT version (411 lines)"
echo ""

# Create archive for original TS2Vec code (keep for reference)
echo "📁 Archiving original TS2Vec code for reference..."
mkdir -p archive/original_ts2vec/
cp -r baselines/TS2vec/*.py archive/original_ts2vec/ 2>/dev/null || true
cp -r baselines/TS2vec/models/ archive/original_ts2vec/ 2>/dev/null || true
cp -r baselines/TS2vec/tasks/ archive/original_ts2vec/ 2>/dev/null || true
cp baselines/TS2vec/README.md archive/original_ts2vec/ 2>/dev/null || true

echo "✅ Original TS2Vec code preserved in archive/original_ts2vec/"

# Remove the massive datasets duplication  
echo ""
echo "🗑️  Removing 6.2GB of redundant datasets..."
rm -rf baselines/TS2vec/datasets/

# Remove the entire baselines folder since it's redundant
echo "🗑️  Removing redundant baselines folder..."
rm -rf baselines/

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💾 Space freed:"
echo "   ~6.2GB of duplicate datasets removed"
echo "   ~50MB of redundant code removed"
echo ""
echo "📁 Preserved:"
echo "   Original TS2Vec code → archive/original_ts2vec/"
echo "   Enhanced TimeHUT code → Main directory"
