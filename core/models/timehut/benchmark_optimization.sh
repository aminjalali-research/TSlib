#!/bin/bash

# TimeHUT Performance Comparison Script
# Tests original vs optimized TimeHUT implementation

echo "🚀 TimeHUT Performance Optimization Test"
echo "========================================"

DATASET="Chinatown"
EPOCHS=200
BATCH_SIZE=8

echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS" 
echo "Batch Size: $BATCH_SIZE"
echo ""

# Test original TimeHUT
echo "📊 Testing Original TimeHUT..."
echo "⏱️  Starting timer..."
START_TIME=$(date +%s)

cd /home/amin/TimesURL/methods/TimeHUT
python train.py $DATASET original_test --loader UCR --gpu 0 --batch-size $BATCH_SIZE --epochs $EPOCHS --eval

END_TIME=$(date +%s)
ORIGINAL_TIME=$((END_TIME - START_TIME))
echo "✅ Original completed in: ${ORIGINAL_TIME}s"
echo ""

# Test optimized TimeHUT (skip search for small dataset)
echo "🔧 Testing Optimized TimeHUT (Skip Search)..."
echo "⏱️  Starting timer..."
START_TIME=$(date +%s)

python train_optimized.py $DATASET optimized_skip_test --loader UCR --gpu 0 --batch-size $BATCH_SIZE --epochs $EPOCHS --eval --skip-search

END_TIME=$(date +%s)
OPTIMIZED_SKIP_TIME=$((END_TIME - START_TIME))
echo "✅ Optimized (Skip Search) completed in: ${OPTIMIZED_SKIP_TIME}s"
echo ""

# Test optimized TimeHUT (reduced search steps)
echo "⚡ Testing Optimized TimeHUT (3 Search Steps)..."
echo "⏱️  Starting timer..."
START_TIME=$(date +%s)

python train_optimized.py $DATASET optimized_search_test --loader UCR --gpu 0 --batch-size $BATCH_SIZE --epochs $EPOCHS --eval --search-steps 3

END_TIME=$(date +%s)
OPTIMIZED_SEARCH_TIME=$((END_TIME - START_TIME))
echo "✅ Optimized (3 Steps) completed in: ${OPTIMIZED_SEARCH_TIME}s"
echo ""

# Calculate improvements
SKIP_IMPROVEMENT=$(echo "scale=1; (($ORIGINAL_TIME - $OPTIMIZED_SKIP_TIME) * 100) / $ORIGINAL_TIME" | bc)
SEARCH_IMPROVEMENT=$(echo "scale=1; (($ORIGINAL_TIME - $OPTIMIZED_SEARCH_TIME) * 100) / $ORIGINAL_TIME" | bc)

echo "📈 PERFORMANCE COMPARISON RESULTS"
echo "================================"
echo "Original TimeHUT:        ${ORIGINAL_TIME}s"
echo "Optimized (Skip Search): ${OPTIMIZED_SKIP_TIME}s  (${SKIP_IMPROVEMENT}% faster)"
echo "Optimized (3 Steps):     ${OPTIMIZED_SEARCH_TIME}s  (${SEARCH_IMPROVEMENT}% faster)"
echo ""

if (( $(echo "$SKIP_IMPROVEMENT > 40" | bc -l) )); then
    echo "🎉 SUCCESS: Skip Search optimization achieved >40% speedup!"
else
    echo "⚠️  WARNING: Skip Search optimization less than expected"
fi

if (( $(echo "$SEARCH_IMPROVEMENT > 20" | bc -l) )); then
    echo "🎉 SUCCESS: Reduced Search optimization achieved >20% speedup!"
else
    echo "⚠️  WARNING: Reduced Search optimization less than expected"
fi

echo ""
echo "💡 RECOMMENDATIONS:"
echo "- For small datasets (like Chinatown): Use --skip-search for maximum speed"
echo "- For larger datasets: Use --search-steps 3-5 for good balance"
echo "- Monitor GPU memory with: nvidia-smi"

# Generate comparison table
echo ""
echo "📊 DETAILED COMPARISON TABLE"
echo "| Method | Time (s) | Speedup | Use Case |"
echo "|--------|----------|---------|----------|"
echo "| Original | $ORIGINAL_TIME | baseline | Research/benchmarking |"
echo "| Skip Search | $OPTIMIZED_SKIP_TIME | ${SKIP_IMPROVEMENT}% | Small datasets |"
echo "| 3 Steps | $OPTIMIZED_SEARCH_TIME | ${SEARCH_IMPROVEMENT}% | General use |"
