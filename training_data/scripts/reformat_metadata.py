with open("../MyTTSDataset/metadata.csv","r") as f:
    lines = f.readlines()

with open("../MyTTSDataset/metadata.csv","w") as f:
    for line in lines:
        line = line.split("|")
        line = [x.replace("\n","").strip() for x in line] + [line[-1]]
        f.write("|".join(line))