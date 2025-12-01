# ============================================================================
# Standard conditional training (all graphs at once)
# ============================================================================
# CUDA_VISIBLE_DEVICES=0 python problems/graph_coloring/main.py --conditional --graphs myciel2 --max-colors 191 --steps 10000 &
# CUDA_VISIBLE_DEVICES=1 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 --max-colors 191 --steps 10000 &
# CUDA_VISIBLE_DEVICES=2 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 --max-colors 191 --steps 10000 &
# CUDA_VISIBLE_DEVICES=3 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 myciel5 --max-colors 191 --steps 10000 &
# CUDA_VISIBLE_DEVICES=0 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 myciel5 myciel6 --max-colors 191 --steps 10000 &
# CUDA_VISIBLE_DEVICES=1 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 myciel5 myciel6 myciel7 --max-colors 191 --steps 10000 &

# ============================================================================
# Curriculum learning (gradually add harder graphs: myciel2 -> 3 -> 4 -> 5 -> 6 -> 7)
# ============================================================================
# Full curriculum (all stages)
CUDA_VISIBLE_DEVICES=0 python problems/graph_coloring/train_curriculum.py --steps-per-stage 5000 --max-colors 191 &

# Partial curriculum (up to myciel5)
# CUDA_VISIBLE_DEVICES=1 python problems/graph_coloring/train_curriculum.py --stages myciel2 myciel3 myciel4 myciel5 --steps-per-stage 5000 --max-colors 191 &