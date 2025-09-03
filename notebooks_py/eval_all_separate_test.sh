# echo "Evaluating gpt-5-mini Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gpt-5-mini
# echo "Evaluating claude-sonnet-4-20250514 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models claude-sonnet-4-20250514
# echo "Evaluating gemini-2.5-pro Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gemini-2.5-pro
echo "Evaluating llava-hf/llava-v1.6-mistral-7b-hf Cell Selection"
SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models llava-hf/llava-v1.6-mistral-7b-hf
echo "Evaluating Qwen/Qwen2.5-VL-7B-Instruct Cell Selection"
SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models Qwen/Qwen2.5-VL-7B-Instruct
echo "Evaluuating mistralai/Pixtral-12B-2409 Cell Selection"
SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models mistralai/Pixtral-12B-2409
echo "Evaluating deepseek-ai/deepseek-vl2 Cell Selection"
SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models deepseek-ai/deepseek-vl2

# echo "Evaluating gpt-5-mini Pointing"
# SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gpt-5-mini
# echo "Evaluating claude-sonnet-4-20250514 Pointing"
# SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models claude-sonnet-4-20250514
# echo "Evaluating gemini-2.5-pro Pointing"
# SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gemini-2.5-pro
echo "Evaluating llava-hf/llava-v1.6-mistral-7b-hf Pointing"
SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models llava-hf/llava-v1.6-mistral-7b-hf
echo "Evaluating Qwen/Qwen2.5-VL-7B-Instruct Pointing"
SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models Qwen/Qwen2.5-VL-7B-Instruct
echo "Evaluuating mistralai/Pixtral-12B-2409 Pointing"
SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models mistralai/Pixtral-12B-2409
echo "Evaluating deepseek-ai/deepseek-vl2 Pointing"
SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models deepseek-ai/deepseek-vl2
