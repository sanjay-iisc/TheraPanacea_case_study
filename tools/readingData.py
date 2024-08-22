import glob
def getTrainLabels(path)->list:
    a = []
    with open(path, "r") as f:
        lines=f.readlines()
        for numLine,line in enumerate(lines):
            print(f"Line {numLine}: ", line)
            a.append(line)
    assert len(a)==1e5
    return a;

def getTrainDataPathList(train_path)->list:
    train_images=glob.glob("{:s}/*.jpg".format(train_path))
    
    assert len(train_images)==1e5, "Size of Train images is not equal to 1e5"
    return  sorted(train_images)

    
if __name__=="__main__":
    a=getTrainLabels("./data/label_train.txt")
    b=getTrainDataPathList("./data/train_img")
    for index, num in enumerate(a):
        if index==1597:
            
            print(b[index],index, num)
    print(len(a)==len(b))


