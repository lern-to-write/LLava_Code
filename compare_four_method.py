import json

def load_data(file_path):
    """加载JSON Lines文件并构建doc_id索引"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return {str(item["doc_id"]): item for item in (json.loads(line) for line in f)}

def find_file_errors(file1, file2, file3, file4):
    # 加载四个文件的数据
    data1 = load_data(file1)
    data2 = load_data(file2)
    data3 = load_data(file3)
    data4 = load_data(file4)
    
    # 获取四个文件共有的doc_id
    common_ids = set(data1.keys()) & set(data2.keys()) & set(data3.keys()) & set(data4.keys())
    
    results = []
    for doc_id in common_ids:
        item1 = data1[doc_id]
        item2 = data2[doc_id]
        item3 = data3[doc_id]
        item4 = data4[doc_id]
        
        # 检查文件1是否正确
        is_file1_correct = item1["mvbench_accuracy"]["pred_answer"] == item1["mvbench_accuracy"]["gt_answer"]
        # 检查其他文件是否错误
        is_file2_wrong = item2["mvbench_accuracy"]["pred_answer"] != item2["mvbench_accuracy"]["gt_answer"]
        is_file3_wrong = item3["mvbench_accuracy"]["pred_answer"] != item3["mvbench_accuracy"]["gt_answer"]
        is_file4_wrong = item4["mvbench_accuracy"]["pred_answer"] != item4["mvbench_accuracy"]["gt_answer"]
        
        if is_file1_correct and is_file2_wrong and is_file3_wrong and is_file4_wrong:
            results.append({
                "doc_id": doc_id,
                "video": item1["doc"]["video"],
                "question": item1["doc"]["question"],
                "correct_answer": item1["mvbench_accuracy"]["gt_answer"],
                "file1_pred": item1["mvbench_accuracy"]["pred_answer"],
                "file2_pred": item2["mvbench_accuracy"]["pred_answer"],
                "file3_pred": item3["mvbench_accuracy"]["pred_answer"],
                "file4_pred": item4["mvbench_accuracy"]["pred_answer"]
            })
    return results

if __name__ == "__main__":
    file1_path = "/obs/users/yiyu/new_env2/vidcom2_15/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_215053_samples_mvbench_state_change.jsonl"
    file2_path = "/obs/users/yiyu/new_env2/pdrop_0.15/lmms-lab__llava-onevision-qwen2-7b-ov/20250520_052138_samples_mvbench_state_change.jsonl"
    file3_path = "/obs/users/yiyu/mvbench/home/chunkui.mjp/code/LLaVA-NeXT_sparsevlm/logs/vision_model__llava-onevision-qwen2-7b-ov/20250519_213225_samples_mvbench_state_change.jsonl"
    file4_path = "/obs/users/yiyu/new_env2/fastv_15/lmms-lab__llava-onevision-qwen2-7b-ov/20250520_001532_samples_mvbench_state_change.jsonl"  # 请替换为实际路径
    
    results = find_file_errors(file1_path, file2_path, file3_path, file4_path)
    
    with open('four_files_analysis_results.txt', 'w', encoding='utf-8') as f:
        print(f"找到 {len(results)} 个符合条件的问题:", file=f)
        print(f"找到 {len(results)} 个符合条件的问题:")
        
        for idx, r in enumerate(results, 1):
            output = f"""
{idx}. 文档ID: {r['doc_id']}
   Video文件: {r['video']}
   问题: {r['question']}
   正确答案: {r['correct_answer']}
   文件1预测: {r['file1_pred']}
   文件2预测: {r['file2_pred']}
   文件3预测: {r['file3_pred']}
   文件4预测: {r['file4_pred']}
"""
            print(output, file=f)
            print(output)