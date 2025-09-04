# This script integrates model inference with performance evaluation.
# It connects to a running Qwen2.5-VL model server to get responses for a dataset,
# and then evaluates the generated text against reference responses.

import json
import os
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import classification_report
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Download necessary NLTK data for tokenization.
# This will automatically download the 'punkt' resource if it's not present.
nltk.download('punkt')

# --- Model API and Data Configuration ---

# IMPORTANT: This is the address of your running Qwen2.5-VL model server.
MODEL_API_URL = "http://10.10.0.1:8000/v1/chat/completions"

# IMPORTANT: Replace this with the actual path to your dataset JSON file.
DATASET_FILE_PATH = os.path.join("/home/yinglanliang/LLaMA-Factory/data/mllm_data", "mllm_data.json")

# --- Utility Functions ---

def get_data(file_path):
    """
    Loads the dataset from a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件未找到: '{file_path}'。请确保文件存在且路径正确。")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据集文件: '{file_path}'")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"无法解析 JSON 文件 '{file_path}'。请检查文件格式是否正确。错误: {e}", e.doc, e.pos)

def load_image_from_path_and_base64(image_path):
    """
    Loads an image from a local file path and converts it to a base64 string.
    """
    full_image_path = os.path.join(os.path.dirname(DATASET_FILE_PATH), image_path)
    if not os.path.exists(full_image_path):
        print(f"警告: 图像文件未找到: {full_image_path}. 跳过此样本。")
        return None
    
    with open(full_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string

# --- Model Inference Function via API Call ---

def get_response_from_api(api_url, image_b64, prompt):
    """
    Sends a request to the model API to get a generated response.
    """
    headers = {"Content-Type": "application/json"}
    
    # Construct the messages payload in the format the server expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    payload = {
        "model": "qwen2.5-vl", # This might need to be adjusted based on your server
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=600)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Assuming the API returns a JSON object with a 'content' key
        generated_text = response.json()['choices'][0]['message']['content']
        return generated_text.strip()
    except requests.exceptions.RequestException as e:
        print(f"调用模型API时发生错误: {e}. 返回空字符串。")
        return ""
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"解析API响应时发生错误: {e}. 返回空字符串。")
        return ""


# --- Evaluation Functions ---

def evaluate_with_rouge(generated, reference):
    print("--- 正在使用 ROUGE-score 进行评估 ---")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = [scorer.score(ref, gen) for gen, ref in zip(generated, reference)]
    
    avg_scores = {
        'rouge1_fmeasure': np.mean([s['rouge1'].fmeasure for s in all_scores]),
        'rouge2_fmeasure': np.mean([s['rouge2'].fmeasure for s in all_scores]),
        'rougeL_fmeasure': np.mean([s['rougeL'].fmeasure for s in all_scores])
    }
    print(json.dumps(avg_scores, indent=4, ensure_ascii=False))
    print("-" * 35)
    return avg_scores


def evaluate_with_bertscore(generated, reference):
    print("--- 正在使用 BERT-score 进行评估 ---")
    # Determine the device to use (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"BERT-score 评估将在设备上运行: {device}")
    
    P, R, F1 = score(generated, reference, lang="en", verbose=False, device=device)
    bert_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
    print(f"平均精确度 (Average Precision): {bert_scores['precision']:.4f}")
    print(f"平均召回率 (Average Recall): {bert_scores['recall']:.4f}")
    print(f"平均 F1 分数 (Average F1 Score): {bert_scores['f1']:.4f}")
    print("-" * 35)
    return bert_scores


def evaluate_with_nltk_bleu(generated, reference):
    """
    使用 NLTK 评估 BLEU 分数，采用平滑函数处理零重叠问题。
    
    Args:
        generated (list): 模型生成的文本列表。
        reference (list): 参考文本列表。
        
    Returns:
        dict: 包含平均 BLEU 分数的字典。
    """
    print("--- 正在使用 NLTK (BLEU Score) 进行评估 ---")

    # 检查输入列表长度是否一致
    if len(generated) != len(reference):
        print("错误：生成的文本数量与参考文本数量不一致，无法进行评估。")
        return {'bleu_score': 0.0}
    
    # 将文本数据转换为 NLTK 的标记化格式
    # corpus_bleu 需要列表格式的参考文本
    references_tokens = [[nltk.word_tokenize(text.lower())] for text in reference]
    generated_tokens = [nltk.word_tokenize(text.lower()) for text in generated]

    # 定义平滑函数，用于处理 n-gram 零重叠的情况
    # Method4 通常能提供更好的平滑效果，适合处理短句或零重叠问题
    smooth_function = SmoothingFunction().method4

    # 使用 corpus_bleu 一次性计算整个语料库的平均 BLEU 分数
    # 这比逐句计算再求平均更准确，因为它会利用整个语料库的统计信息
    # weights=(0.25, 0.25, 0.25, 0.25) 表示计算 1-gram 到 4-gram
    avg_bleu = corpus_bleu(references_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_function)
    
    print(f"平均 BLEU 分数 (Average BLEU Score): {avg_bleu:.4f}")
    print("-" * 35)
    return {'bleu_score': avg_bleu}

def conceptual_evaluate_with_sklearn(generated_texts, reference_texts):
    """
    使用 scikit-learn 对生成的文本进行概念性分类评估。
    
    Args:
        generated_texts (list): 模型生成的文本列表。
        reference_texts (list): 包含正确答案的参考文本列表。
        
    Returns:
        dict: 包含分类报告的字典，如果输入不一致则返回 None。
    """
    print("--- 正在使用 scikit-learn 进行概念性评估 ---")
    
    # 检查输入列表长度是否一致
    if len(generated_texts) != len(reference_texts):
        print("错误：生成的文本数量与参考文本数量不一致！")
        return None

    # 从 reference_texts 中动态提取 true_labels
    true_labels = []
    for ref_text in reference_texts:
        ref_text_lower = ref_text.lower()
        if 'hemorrhage' in ref_text_lower:
            true_labels.append('brain hemorrhage')
        elif 'fracture' in ref_text_lower:
            true_labels.append('distal radius fracture')
        elif 'benign' in ref_text_lower:
            true_labels.append('benign lesion')
        else:
            true_labels.append('other')

    # 生成预测标签
    predicted_labels = []
    for gen_text in generated_texts:
        gen_text_lower = gen_text.lower()
        if 'hemorrhage' in gen_text_lower:
            predicted_labels.append('brain hemorrhage')
        elif 'fracture' in gen_text_lower:
            predicted_labels.append('distal radius fracture')
        elif 'benign' in gen_text_lower:
            predicted_labels.append('benign lesion')
        else:
            predicted_labels.append('other')
            
    # 执行分类报告
    report = classification_report(true_labels, predicted_labels, zero_division=0, output_dict=True)
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print("注意：scikit-learn 部分是用于分类任务的概念性示例。")
    print("对于文本生成任务，ROUGE、BERTScore 或 NLTK 的指标更合适。")
    print("-" * 35)
    return report

# --- Save Results Functions ---

def save_results_as_json(metrics, file_path="evaluation_results.json"):
    """
    Saves evaluation metrics to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"评估结果已保存到文件: {file_path}")

def save_results_as_image(metrics, file_path="evaluation_results.png"):
    """
    Saves evaluation metrics as an image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    results_text = "模型性能评估结果\n\n"
    
    # ROUGE Scores
    results_text += "ROUGE Scores:\n"
    results_text += f"  ROUGE-1 F-measure: {metrics['rouge']['rouge1_fmeasure']:.4f}\n"
    results_text += f"  ROUGE-2 F-measure: {metrics['rouge']['rouge2_fmeasure']:.4f}\n"
    results_text += f"  ROUGE-L F-measure: {metrics['rouge']['rougeL_fmeasure']:.4f}\n\n"
    
    # BERTScore
    results_text += "BERTScore:\n"
    results_text += f"  Average Precision: {metrics['bert']['precision']:.4f}\n"
    results_text += f"  Average Recall: {metrics['bert']['recall']:.4f}\n"
    results_text += f"  Average F1 Score: {metrics['bert']['f1']:.4f}\n\n"
    
    # NLTK BLEU
    results_text += "NLTK BLEU Score:\n"
    results_text += f"  Average BLEU Score: {metrics['nltk']['bleu_score']:.4f}\n\n"
    
    # Scikit-learn (Conceptual)
    results_text += "Scikit-learn (Conceptual):\n"
    for label, data in metrics['sklearn'].items():
        if isinstance(data, dict):
            results_text += f"  {label.capitalize()}:\n"
            results_text += f"    Precision: {data['precision']:.4f}\n"
            results_text += f"    Recall: {data['recall']:.4f}\n"
            results_text += f"    F1-Score: {data['f1-score']:.4f}\n"

    ax.text(0.1, 0.9, results_text, transform=ax.transAxes, fontsize=14, va='top', ha='left',
            family='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.5))
    
    ax.set_title("Model Performance Evaluation Results", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"评估结果图已保存到文件: {file_path}")


# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Load the dataset
        data = get_data(DATASET_FILE_PATH)
        
        generated_responses = []
        reference_responses = []
        
        print("\n--- 正在遍历数据集并生成响应 ---")
        
        # Limit to the first 1000 data points
        for i, item in enumerate(data[:1000]):
            try:
                image_path = item['images'][0]
                user_prompt = item['messages'][0]['content'].replace("<image>", "").strip()
                reference_text = item['messages'][1]['content']
                
                # Load image and convert to Base64
                image_b64 = load_image_from_path_and_base64(image_path)
                if image_b64 is None:
                    continue
                
                print(f"正在为样本 {i+1} 生成响应...")
                generated_text = get_response_from_api(MODEL_API_URL, image_b64, user_prompt)
                
                generated_responses.append(generated_text)
                reference_responses.append(reference_text)
                
            except Exception as e:
                print(f"处理样本 {i+1} 时发生错误: {e}. 跳过。")
                continue

        # Perform evaluations if there are results
        if generated_responses:
            print("\n--- 所有响应已生成。开始评估 ---")
            
            all_metrics = {}
            all_metrics['rouge'] = evaluate_with_rouge(generated_responses, reference_responses)
            all_metrics['bert'] = evaluate_with_bertscore(generated_responses, reference_responses)
            all_metrics['nltk'] = evaluate_with_nltk_bleu(generated_responses, reference_responses)
            all_metrics['sklearn'] = conceptual_evaluate_with_sklearn(generated_responses, reference_responses)
            
            # Save results to a file and as an image
            save_results_as_json(all_metrics)
            save_results_as_image(all_metrics)
            
        else:
            print("未能生成任何响应，无法进行评估。")

    except Exception as e:
        print(f"发生致命错误: {e}")
