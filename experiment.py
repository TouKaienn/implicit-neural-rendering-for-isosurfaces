def GetTextValue(text_path, idx):
    res = []
    with open(text_path, 'r') as f:
        for line in f:
            res.append(list(map(float, line.strip().split(' '))))
    return res[idx]

if __name__ == "__main__":

    for idx in range(0, 5):
        print(GetTextValue(text_path='.\\tiny_vorts0008_normalize_dataset\\vorts0008_infos.txt',
                           idx = idx))
