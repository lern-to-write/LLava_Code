import json

def load_data(file_path):
    """加载JSON Lines文件并构建doc_id索引"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return {str(item["doc_id"]): item for item in (json.loads(line) for line in f)}

def find_three_file_errors(file1, file2, file3):
    # 加载三个文件的数据
    data1 = load_data(file1)
    data2 = load_data(file2)
    data3 = load_data(file3)
    
    # 获取三个文件共有的doc_id
    common_ids = set(data1.keys()) & set(data2.keys()) & set(data3.keys())
    
    results = []
    for doc_id in common_ids:
        item1, item2, item3 = data1[doc_id], data2[doc_id], data3[doc_id]
        
        # 检查文件1和文件3是否都正确
        is_file1_correct = item1["mvbench_accuracy"]["pred_answer"] == item1["mvbench_accuracy"]["gt_answer"]
        is_file3_correct = item3["mvbench_accuracy"]["pred_answer"] == item3["mvbench_accuracy"]["gt_answer"]
        
        # 检查文件2是否错误
        is_file2_wrong = item2["mvbench_accuracy"]["pred_answer"] != item2["mvbench_accuracy"]["gt_answer"]
        
        if is_file1_correct and is_file3_correct and is_file2_wrong:
            results.append({
                "doc_id": doc_id,
                "video": item1["doc"]["video"],
                "question": item1["doc"]["question"],
                "correct_answer": item1["mvbench_accuracy"]["gt_answer"],
                "file1_pred": item1["mvbench_accuracy"]["pred_answer"],
                "file2_pred": item2["mvbench_accuracy"]["pred_answer"],
                "file3_pred": item3["mvbench_accuracy"]["pred_answer"]
            })
    return results

if __name__ == "__main__":
    file1_path = "/obs/users/yiyu/new_env2/full_tokens/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_193630_samples_mvbench_scene_transition.jsonl"
    file2_path = "/obs/users/yiyu/new_env2/8_BIG/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_203513_samples_mvbench_scene_transition.jsonl"
    file3_path = "/obs/users/yiyu/new_env2/24_SMALL/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_200758_samples_mvbench_scene_transition.jsonl"
    results = find_three_file_errors(file1_path, file2_path, file3_path)
    
    with open('big_and_full_analysis_results.txt', 'w', encoding='utf-8') as f:
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
"""
            print(output, file=f)
            print(output)