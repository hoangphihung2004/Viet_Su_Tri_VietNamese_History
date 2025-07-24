from services.rag_service import RAGService
import time
import tqdm
import json


def evaluation(rag, data):
    result = []

    for question in tqdm.tqdm(data, desc="Evaluating"):
        print("Question:", question.get("Question"))
        start = time.time()

        ans = rag.handling_query(user_query=question.get("Question"))
        print("Answer:", ans)

        end = time.time()

        url = ans.get("URL")
        if isinstance(url, list) or url is None:
            final_url = url
        else:
            final_url = [url]

        result.append({
            "Question": question.get("Question"),
            "Ground_Truth": question.get("Ground_Truth"),
            "URL": final_url,
            "Time": round(end - start, 3)
        })

        time.sleep(15)

    return result


def main():
    rag = RAGService()

    with open(r"D:\HistoryVietNam\system_evaluation\ground_truth.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    result = evaluation(rag, data)

    with open(r"D:\HistoryVietNam\system_evaluation\result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    precision_list = []
    recall_list = []

    for item in result:
        gt = item["Ground_Truth"]
        pred = item["URL"]
        if gt is None or pred is None:
            continue
        gt_set = set(gt)
        pred_set = set(pred)
        true_positive = len(gt_set & pred_set)
        precision = true_positive / len(pred_set) if pred_set else 0
        recall = true_positive / len(gt_set) if gt_set else 0
        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_response_time = sum(val["Time"] for val in result) / len(result)

    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average Response Time: {avg_response_time:.3f} seconds")


if __name__ == "__main__":
    main()
