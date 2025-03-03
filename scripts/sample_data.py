import hashlib
import json
import random

def read_data(file_path, n_meta_cols):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            data.append((cols[n_meta_cols], cols[n_meta_cols + 1:]))
    return data

def filter_data(data):
    filtered_data = []
    for src, tgt in data:
        filtered_tgt = []
        for t in tgt:
            if 0.66 * len(src) <= len(t) <= 1.5 * len(src):
                filtered_tgt.append(t)
        if len(filtered_tgt) > 0:
            filtered_data.append((src, filtered_tgt))
    return filtered_data

def sample_data(data, n_samples):
    return random.sample(data, min(n_samples, len(data)))

if __name__ == "__main__":
    random.seed(42)
    # handle lang8 dataset
    lang8_data = read_data("datasets/lang8/train.txt", 2)
    lang8_data = filter_data(lang8_data)
    print(len(lang8_data))
    lang8_data = sample_data(lang8_data, 10000)

    # handle cctc dataset
    cctc_data = read_data("datasets/cctc/train.para", 1)
    cctc_data = filter_data(cctc_data)
    print(len(cctc_data))
    cctc_data = sample_data(cctc_data, 10000)

    # handle fcgec dataset
    fcgec_data = read_data("datasets/fcgec/fcgec.train.offical-filtered.para", 1)
    fcgec_data = filter_data(fcgec_data)
    print(len(fcgec_data))
    fcgec_data = sample_data(fcgec_data, 10000)

    # handle cscd dataset
    cscd_data = read_data("datasets/cscd/train.txt", 0)
    cscd_data = filter_data(cscd_data)
    print(len(cscd_data))
    cscd_data = sample_data(cscd_data, 10000)

    data = {
        "lang8": lang8_data,
        "cctc": cctc_data,
        "fcgec": fcgec_data,
        "cscd": cscd_data,
    }

    with open("datasets/train_split/data.jsonl", "w") as f:
        for k, v in data.items():
            for src, tgt in v:
                md5_id = hashlib.md5(f"{k}:{src}".encode()).hexdigest()
                f.write(json.dumps({
                    "id": md5_id,
                    "input_text": src,
                    "references": tgt,
                    "dataset": k,
                }, ensure_ascii=False) + "\n")
