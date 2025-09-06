# echo "Evaluating gpt-4.1 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gpt-4.1
# echo "Evaluating claude-sonnet-4-20250514 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models claude-sonnet-4-20250514
# echo "Evaluating gemini-2.0-flash Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models gemini-2.0-flash
# echo "Evaluating llava-hf/llava-v1.6-mistral-7b-hf Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models llava-hf/llava-v1.6-mistral-7b-hf
# echo "Evaluating Qwen/Qwen2.5-VL-7B-Instruct Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models Qwen/Qwen2.5-VL-7B-Instruct
# echo "Evaluuating mistralai/Pixtral-12B-2409 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models mistralai/Pixtral-12B-2409
# echo "Evaluating deepseek-ai/deepseek-vl2 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models deepseek-ai/deepseek-vl2
# echo "Evaluating nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50 Cell Selection"
# SKIP_POINTING=true ./eval_both_advanced.sh --samples 1 --no-persistent --models nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50

# echo "Evaluating gpt-4.1 Pointing"
# ./eval_both_advanced.sh --samples 1 --models gpt-4.1 --skip-cell
# echo "Evaluating claude-sonnet-4-20250514 Pointing"
# ./eval_both_advanced.sh --samples 1 --models claude-sonnet-4-20250514 --skip-cell
# echo "Evaluating gemini-2.0-flash Pointing"
# ./eval_both_advanced.sh --samples 1 --models gemini-2.0-flash --skip-cell
echo "Evaluating llava-hf/llava-v1.6-mistral-7b-hf Pointing"
./eval_both_advanced.sh --samples 100 --models llava-hf/llava-v1.6-mistral-7b-hf --skip-cell
echo "Evaluating Qwen/Qwen2.5-VL-7B-Instruct Pointing"
./eval_both_advanced.sh --samples 100 --models Qwen/Qwen2.5-VL-7B-Instruct --skip-cell
echo "Evaluuating mistralai/Pixtral-12B-2409 Pointing"
./eval_both_advanced.sh --samples 100 --models mistralai/Pixtral-12B-2409 --skip-cell
# echo "Evaluating deepseek-ai/deepseek-vl2 Pointing"
# SKIP_CELL=true ./eval_both_advanced.sh --samples 1 --no-persistent --models deepseek-ai/deepseek-vl2
echo "Evaluating nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50 Pointing"
./eval_both_advanced.sh --samples 100 --models nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50 --skip-cell
