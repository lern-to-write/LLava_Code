import json

def load_data(file_path):
    """加载JSON Lines文件并构建问题ID索引"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return {item["doc"]["question_id"]: item for item in (json.loads(line) for line in f)}

def find_three_file_errors(file1, file2, file3):
    # 加载三个文件的数据
    data1 = load_data(file1)
    data2 = load_data(file2)
    data3 = load_data(file3)
    
    # 获取三个文件共有的question_id
    common_qids = set(data1.keys()) & set(data2.keys()) & set(data3.keys())
    
    results = []
    for qid in common_qids:
        item1 = data1[qid]
        item2 = data2[qid]
        item3 = data3[qid]
        
        # 检查文件1和文件3是否都正确
        is_file1_correct = item1["videomme_perception_score"]["pred_answer"] == item1["doc"]["answer"]
        is_file3_correct = item3["videomme_perception_score"]["pred_answer"] == item3["doc"]["answer"]
        
        # 检查文件2是否错误
        is_file2_wrong = item2["videomme_perception_score"]["pred_answer"] != item2["doc"]["answer"]
        
        # if is_file1_correct and is_file3_correct and is_file2_wrong:
        if is_file1_correct  and is_file2_wrong:

            results.append({
                "question_id": qid,
                "videoID": item1["doc"]["videoID"],
                "question": item1["doc"]["question"],
                "correct_answer": item1["doc"]["answer"],
                "file1_pred": item1["videomme_perception_score"]["pred_answer"],
                "file2_pred": item2["videomme_perception_score"]["pred_answer"],
                "file3_pred": item3["videomme_perception_score"]["pred_answer"]
            })
    return results

if __name__ == "__main__":
    # 实际文件路径（已修复重复的main块问题）
    file1_path = "/obs/users/yiyu/new_env2/full_tokens/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_193630_samples_mvbench_action_antonym.jsonl"
    file2_path = "/obs/users/yiyu/new_env2/8_BIG/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_203513_samples_mvbench_action_antonym.jsonl"
    file3_path = "/obs/users/yiyu/new_env2/24_SMALL/lmms-lab__llava-onevision-qwen2-7b-ov/20250519_200758_samples_mvbench_action_antonym.jsonl"
    
    results = find_three_file_errors(file1_path, file2_path, file3_path)
    
    with open('big_and_full_analysis_results.txt', 'w', encoding='utf-8') as f:  # 新增文件写入
        # 打印到控制台和文件
        print(f"找到 {len(results)} 个符合条件的问题:", file=f)
        print(f"找到 {len(results)} 个符合条件的问题:")
        
        for idx, r in enumerate(results, 1):
            # 构建输出内容
            output = f"""
{idx}. 问题ID: {r['question_id']}
   VideoID: {r['videoID']}
   问题: {r['question']}
   正确答案: {r['correct_answer']}
   文件1预测: {r['file1_pred']}
   文件2预测: {r['file2_pred']}
   文件3预测: {r['file3_pred']}
"""
            # 写入文件和控制台
            print(output, file=f)
            print(output)