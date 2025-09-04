import os
import json
import zipfile
import requests
import numpy as np
import base64
from typing import List, Union

# --- 配置 ---
API_URL = "http://localhost:8001/v1/embeddings"
DATASET_PATH = "./"
OUTPUT_DIR_NPY = "output"
OUTPUT_DIR_TXT = "output_txt"
MAX_FILE_SIZE_MB = 10 

def get_embedding(input_data: Union[str, List[str]]):
    """
    通过 API 调用来获取嵌入向量。
    """
    if isinstance(input_data, str):
        input_list = [input_data]
    else:
        input_list = input_data

    payload = {
        "input": input_list,
        "model": "blip2-embed"
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        embeddings_data = response.json()['data']
        embeddings = [item['embedding'] for item in embeddings_data]
        return embeddings
    except requests.exceptions.RequestException as e:
        print(f"API 调用失败: {e}")
        return None

def save_and_split_vectors_npy(vectors: List[List[float]], output_filename: str):
    """
    将向量保存到 .npy 文件，并根据文件大小进行分割。
    """
    if not vectors:
        print("没有向量数据可保存。")
        return

    vectors_np = np.array(vectors)
    if vectors_np.size == 0:
        print("向量数组为空，没有数据可保存。")
        return

    vector_size_bytes = 768 * vectors_np.itemsize
    
    current_file_index = 1
    current_file_path = os.path.join(OUTPUT_DIR_NPY, f"{output_filename}_{current_file_index}.npy")
    
    current_size_bytes = 0
    start_index = 0
    
    os.makedirs(OUTPUT_DIR_NPY, exist_ok=True)
    print(f"开始保存向量到 {OUTPUT_DIR_NPY} 目录...")
    
    for i in range(vectors_np.shape[0]):
        row_bytes = vector_size_bytes
        
        if current_size_bytes + row_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
            batch_to_save = vectors_np[start_index:i]
            np.save(current_file_path, batch_to_save)
            print(f"已保存文件: {current_file_path}, 包含 {batch_to_save.shape[0]} 个向量。")
            
            current_file_index += 1
            current_file_path = os.path.join(OUTPUT_DIR_NPY, f"{output_filename}_{current_file_index}.npy")
            current_size_bytes = 0
            start_index = i
            
        current_size_bytes += row_bytes

    final_batch = vectors_np[start_index:]
    if final_batch.shape[0] > 0:
        np.save(current_file_path, final_batch)
        print(f"已保存最终文件: {current_file_path}, 包含 {final_batch.shape[0]} 个向量。")


def convert_npy_to_txt(input_dir, output_dir):
    """
    将指定目录下的所有 .npy 文件转换为 .txt 文件，并按大小分割。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            npy_path = os.path.join(input_dir, filename)
            
            base_filename = filename.replace(".npy", "")
            current_txt_file_index = 1
            current_txt_file_path = os.path.join(output_dir, f"{base_filename}_{current_txt_file_index}.txt")
            
            try:
                # 1. 加载 .npy 文件
                data = np.load(npy_path)
                
                print(f"\n开始转换 {npy_path}...")
                
                # 2. 逐行写入并分割
                current_size_bytes = 0
                f = open(current_txt_file_path, 'w', encoding='utf-8')
                
                for row in data:
                    row_str = ' '.join(map(str, row)) + '\n'
                    row_size_bytes = len(row_str.encode('utf-8'))
                    
                    if current_size_bytes + row_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                        f.close()
                        print(f"已保存文件: {current_txt_file_path}")
                        
                        current_txt_file_index += 1
                        current_txt_file_path = os.path.join(output_dir, f"{base_filename}_{current_txt_file_index}.txt")
                        current_size_bytes = 0
                        f = open(current_txt_file_path, 'w', encoding='utf-8')

                    f.write(row_str)
                    current_size_bytes += row_size_bytes

                f.close()
                print(f"已保存最终文件: {current_txt_file_path}")
                
            except Exception as e:
                print(f"转换文件失败: {npy_path}, 错误: {e}")


def process_dataset():
    """主函数，处理数据集并生成嵌入向量。"""
    
    print("--- 第一步：解压文件和读取数据 ---")
    
    images_dir = os.path.join(DATASET_PATH, "images")
    if not os.path.exists(images_dir):
        with zipfile.ZipFile(os.path.join(DATASET_PATH, 'images.zip'), 'r') as zip_ref:
            zip_ref.extractall(DATASET_PATH)
        print("图像文件已解压。")
    
    try:
        with open(os.path.join(DATASET_PATH, 'Descriptions.json'), 'r') as f:
            descriptions_data = json.load(f)
    except FileNotFoundError:
        print("错误: Descriptions.json 未找到。请确保它在同一目录下。")
        return
    except json.JSONDecodeError:
        print("错误: 无法解析 Descriptions.json。文件可能为空或已损坏。")
        return

    print("--- 第二步：生成文本嵌入向量 ---")
    all_text_embeddings = []
    for i, item in enumerate(descriptions_data):
        try:
            caption = item['Description']['Caption']
            embedding = get_embedding(caption)
            if embedding:
                all_text_embeddings.extend(embedding)
                print(f"已处理第 {i+1} 条文本。")
        except KeyError as e:
            print(f"错误: 无法在数据中找到 {e} 键。")
            continue
        except Exception as e:
            print(f"处理第 {i+1} 条文本失败: {e}")
            continue

    if all_text_embeddings:
        save_and_split_vectors_npy(all_text_embeddings, "text_embeddings")
    
    print("--- 第三步：生成图像嵌入向量 ---")
    
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    all_image_embeddings = []
    
    for i, path in enumerate(image_paths):
        try:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                embedding = get_embedding(f"data:image/jpeg;base64,{encoded_string}")
                if embedding:
                    all_image_embeddings.extend(embedding)
                    print(f"已处理 {i+1} / {len(image_paths)} 张图像。")
        except Exception as e:
            print(f"处理图像 {path} 失败: {e}")
            
    if all_image_embeddings:
        save_and_split_vectors_npy(all_image_embeddings, "image_embeddings")
        
    print("\n--- 任务完成 ---")
    print(f"所有 .npy 向量文件已保存到 ./{OUTPUT_DIR_NPY} 目录。")
    
    # 新增的转换步骤
    print("\n--- 第四步：将 .npy 文件转换为 .txt ---")
    convert_npy_to_txt(OUTPUT_DIR_NPY, OUTPUT_DIR_TXT)
    print("所有向量文件已成功转换并保存到 ./{OUTPUT_DIR_TXT} 目录。")


if __name__ == "__main__":
    from PIL import Image
    process_dataset()
